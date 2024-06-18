import torch
import wandb
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import defaultdict
from optimizers import ExtraAdam

def compute_log_probs(logits, labels, prompt_len, pad_token_id):
    labels = labels[:, prompt_len + 1:].clone()
    logits = logits[:, prompt_len:-1, :] 
    loss_mask = (labels != pad_token_id)

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    return (per_token_logps * loss_mask).sum(-1)

def concat_sequences(seq1, seq2, seq_len):
        truncate_len = min(seq1['input_ids'].size()[1], seq2['input_ids'].size()[1], seq_len)
        input_ids = torch.cat((seq1['input_ids'][:, :truncate_len], seq2['input_ids'][:, :truncate_len]), dim=0)
        attention_mask = torch.cat((seq1['attention_mask'][:, :truncate_len], seq2['attention_mask'][:, :truncate_len]), dim=0)
        return input_ids, attention_mask


def preference_loss(model, input_ids, attention_mask, return_outputs=False):
    rewards = model(
        input_ids=input_ids, attention_mask=attention_mask
    )[0]
    batch_size = input_ids.size(0) // 2 
    rewards_chosen, rewards_rejected= torch.split(rewards, [batch_size, batch_size], dim=0)
    loss = -torch.nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
    if return_outputs:
        return loss, {"rewards_chosen": rewards_chosen, "rewards_rejected": rewards_rejected}
    return loss

def policy_loss(model, input_ids, attention_mask, labels, preference_scores, reference_model, beta, prompt_len, pad_token_id):
    with torch.no_grad():
        ref_logits = reference_model(input_ids=input_ids, attention_mask=attention_mask).logits

    model_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits


    model_log_probs = compute_log_probs(model_logits, labels, prompt_len, pad_token_id)

    # Compute the preference scores, multiply them with the log_probs, and add KL-term
    kl = F.kl_div(F.log_softmax(model_logits, dim=-1), F.softmax(ref_logits, dim=-1), reduction='batchmean', log_target=False)
    loss = preference_scores.view(model_log_probs.shape[0]) * model_log_probs - beta * kl
    return -loss, kl



def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True

def train_main_model(trainer, batch_size, num_batches):
    # Freeze preference model
    freeze_model(trainer.preference_model)
    trainer.model.train()
    
    train_loader = DataLoader(trainer.train_data, batch_size=batch_size, shuffle=True)
    count = 0
    
    for batch in train_loader:
        if num_batches and count == num_batches:
            break 
        count += 1
        
        prompts_encoded = trainer.tokenizer(batch['text'], return_tensors="pt", max_length=trainer.prompt_len, padding=True, truncation=True).to(trainer.devices[0])
        prompts = [trainer.tokenizer.decode(prompt, skip_special_tokens=True) for prompt in prompts_encoded['input_ids']]
        prompts_encoded = {k: v.to(trainer.devices[0]) for k, v in prompts_encoded.items()}
        model_to_use = trainer.model.module if isinstance(trainer.model, torch.nn.DataParallel) else trainer.model
        output_sequences = model_to_use.generate(
            input_ids=prompts_encoded['input_ids'],
            max_length=trainer.sequence_len,
            num_return_sequences=2,
            do_sample=True, 
            pad_token_id=trainer.tokenizer.eos_token_id,
        )
        prompts = [prompt for prompt in prompts for _ in range(2)]
        completions = [trainer.tokenizer.decode(output_sequence, skip_special_tokens=True) for output_sequence in output_sequences]
        completions = [completion[len(prompt):] for prompt, completion in zip(prompts, completions)]
        output_sequences = trainer.tokenizer(completions, return_tensors="pt", padding=True).to(trainer.devices[0])


        with torch.no_grad():
            preference_scores = trainer.preference_model(input_ids=output_sequences['input_ids'], attention_mask=output_sequences['attention_mask'])[0]

        # Model updating 
        loss, kl = policy_loss(trainer.model, output_sequences['input_ids'], output_sequences['attention_mask'], output_sequences['input_ids'], preference_scores, trainer.ref_model, trainer.beta, trainer.prompt_len, trainer.tokenizer.pad_token_id)

        loss = loss.mean()
        loss.backward() 

        params = torch.nn.utils.parameters_to_vector(trainer.model.parameters())
        grads = torch.nn.utils.parameters_to_vector(p.grad for p in trainer.model.parameters() if p.grad is not None)

        param_norm = torch.norm(params, 2).item()
        grad_norm = torch.norm(grads, 2).item()

        trainer.optimizer.step()
        trainer.optimizer.zero_grad() 
        
        wandb.log({
            "policy_eq_check/loss": loss.item(),
            "policy_eq_check/kl": kl.item(),
            "policy_eq_check/param_norm": param_norm, 
            "policy_eq_check/grad_norm": grad_norm,
            "policy_eq_check/iteration": count,
            "policy_eq_check/examples": len(train_loader) * batch_size + count * batch_size,
        })    
    unfreeze_model(trainer.preference_model)

def train_preference_model(trainer, batch_size, num_batches):
    freeze_model(trainer.model)
    trainer.preference_model.train()
    
    train_loader = DataLoader(trainer.train_data, batch_size=batch_size, shuffle=True)
    count = 0
    for batch in train_loader:
        if num_batches and count == num_batches:
            break 
        count += 1

        prompts_encoded = trainer.tokenizer(batch['text'], return_tensors="pt", max_length=trainer.prompt_len, padding=True, truncation=True).to(trainer.devices[0])
        prompts = [trainer.tokenizer.decode(prompt, skip_special_tokens=True) for prompt in prompts_encoded['input_ids']]
        prompts_encoded = {k: v.to(trainer.devices[0]) for k, v in prompts_encoded.items()}
        model_to_use = trainer.model.module if isinstance(trainer.model, torch.nn.DataParallel) else trainer.model
        output_sequences = model_to_use.generate(
            input_ids=prompts_encoded['input_ids'],
            max_length=trainer.sequence_len,
            num_return_sequences=trainer.num_return_sequences,
            do_sample=True, 
            pad_token_id=trainer.tokenizer.eos_token_id,
        )
        # Rank completions with ground-truth / fixed reward model 
        output_sequences = output_sequences.view(batch_size, trainer.num_return_sequences, trainer.sequence_len)
        with torch.no_grad():
            completions_by_prompt = [[trainer.tokenizer.decode(output_sequence, skip_special_tokens=True) for output_sequence in responses] for responses in output_sequences]
            completions_by_prompt = [[c[len(prompt):] for c in completions] for prompt, completions in zip(prompts, completions_by_prompt)]
            rewards_by_prompt = [list(map(lambda c: trainer.reward_func(prompt, c), completions)) for prompt, completions in zip(prompts, completions_by_prompt)]
        
        preference_batch = defaultdict(list)
        for i, prompt in enumerate(prompts):
            preference_batch['prompt'].append(prompt)
            completions = completions_by_prompt[i]
            rewards = rewards_by_prompt[i]
            preference_batch['winning'].append(completions[np.argmax(rewards)])
            preference_batch['losing'].append(completions[np.argmin(rewards)])
            
        # Preference learning 
        good_completions = preference_batch['winning']
        bad_completions = preference_batch['losing']
        good_sequences_encoded = trainer.tokenizer(good_completions, return_tensors="pt", padding=True).to(trainer.devices[0])
        bad_sequences_encoded = trainer.tokenizer(bad_completions, return_tensors="pt", padding=True).to(trainer.devices[0])
        input_ids, attention_mask = concat_sequences(good_sequences_encoded, bad_sequences_encoded, trainer.sequence_len)
        
        # Preference learning 
        pref_loss, rewards = preference_loss(trainer.preference_model, input_ids, attention_mask, return_outputs=True)
        rewards_chosen, rewards_rejected = rewards['rewards_chosen'], rewards['rewards_rejected']
        predictions = (rewards_chosen > rewards_rejected).detach().float()
        pref_loss.backward()

        params = torch.nn.utils.parameters_to_vector(trainer.preference_model.parameters())
        grads = torch.nn.utils.parameters_to_vector(p.grad for p in trainer.preference_model.parameters() if p.grad is not None)

        param_norm = torch.norm(params, 2).item()
        grad_norm = torch.norm(grads, 2).item()

        if isinstance(trainer.preference_optimizer, ExtraAdam):
            trainer.preference_optimizer.extrapolation()
        trainer.preference_optimizer.step()
        trainer.preference_optimizer.zero_grad()

        with torch.no_grad():
            correct_predictions = predictions.sum().item()
            accuracy = correct_predictions / batch_size
            avg_chosen_reward = rewards_chosen.mean().item()
            avg_rejected_reward = rewards_rejected.mean().item()

        wandb.log({
            "pref_eq_check/loss": pref_loss.item(),
            "pref_eq_check/chosen_reward": avg_chosen_reward,
            "pref_eq_check/rejected_reward": avg_rejected_reward,
            "pref_eq_check/accuracy": accuracy,
            "pref_eq_check/margins": avg_chosen_reward - avg_rejected_reward,
            "pref_eq_check/param_norm": param_norm, 
            "pref_eq_check/grad_norm": grad_norm, 
        })
        
        print(f"Inner Loop: Iteration {count}, Pref Loss: {pref_loss.item():.4f}, Chosen Reward: {avg_chosen_reward}, Rejected Reward: {avg_rejected_reward}, Accuracy: {accuracy}")
    
    # Unfreeze main model after training
    unfreeze_model(trainer.model)