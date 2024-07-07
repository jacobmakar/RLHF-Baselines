import torch 
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import evaluate
import numpy as np 
import random 
import wandb
from trainers.train_utils import compute_log_probs
from collections import defaultdict

class IteratedDPOTrainer():
    def __init__(
            self,
            model,
            tokenizer,
            ref_model,
            reward_func,
            optimizer,
            beta,
            train_dataset,
            test_dataset,
            num_test_batches,
            prompt_len,
            sequence_len, 
            seed,
            devices,

    ):
        self.model = model
        self.ref_model = ref_model 
        self.reward_func = reward_func
        self.tokenizer = tokenizer
        self.beta = beta 
        self.train_data = train_dataset
        self.test_data = test_dataset
        self.num_test_batches = num_test_batches
        self.prompt_len = prompt_len
        self.optimizer = optimizer 
        self.sequence_len = sequence_len
        self.seed = seed
        self.devices = devices

    def loss(self, pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l, beta):
        pi_ratio = pi_logps_w - pi_logps_l
        ref_ratio = ref_logps_w - ref_logps_l

        losses = -F.logsigmoid(beta * (pi_ratio - ref_ratio))

        chosen_rewards = beta * (pi_logps_w - ref_logps_w).detach()
        rejected_rewards = beta * (pi_logps_l - ref_logps_l).detach()
       
        return losses.mean(), chosen_rewards, rejected_rewards
            
    def train(self, batch_size, num_batches):                
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        #self.test_data = self.test_data.select(range(self.num_test_batches * batch_size))
        test_loader = DataLoader(self.test_data, batch_size=batch_size)

        self.ref_model.eval()
        self.model.train()
    
        count = 0
        for batch in train_loader: 
            #### EVAL LOOP ####
            if count != 0 and count % 10 == 0:
                avg_kl, avg_reward, policy_text_table = evaluate(test_loader=test_loader, 
                                                                 sequence_len=self.sequence_len, 
                                                                 prompt_len=self.prompt_len, 
                                                                 reward_func=self.reward_func, 
                                                                 tokenizer=self.tokenizer,
                                                                 model=self.model, 
                                                                 ref_model=self.ref_model, 
                                                                 model_device=self.devices[0])
                
                wandb.log({
                "eval/avg_kl": avg_kl,
                "eval/avg_reward": avg_reward,
                "eval/examples": count * batch_size,
                "eval/iteration": count,
                "policy_samples": policy_text_table,
                })
                print(f"Iteration {count}, KL Divergence: {avg_kl:.4f}, Average Reward {avg_reward:.4f}")
            #### END EVAL LOOP ####
            if num_batches and count == num_batches:
                break 
            count += 1
            #### TRAIN LOOP ####
            # Generate n completions from batch of prompts
            prompts_encoded = self.tokenizer(batch['text'], return_tensors="pt", max_length=self.prompt_len, padding=True, truncation=True).to(self.devices[0])
            prompts = [self.tokenizer.decode(prompt, skip_special_tokens=True) for prompt in prompts_encoded['input_ids']]
            prompts_encoded = {k: v.to(self.devices[0]) for k, v in prompts_encoded.items()}
            model_to_use = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
            output_sequences = model_to_use.generate(
                input_ids=prompts_encoded['input_ids'],
                max_length=self.sequence_len,
                num_return_sequences=self.num_return_sequences,
                do_sample=True, 
                pad_token_id=self.tokenizer.eos_token_id,
            )
            # Rank completions with ground-truth / fixed reward model 
            output_sequences = output_sequences.view(batch_size, self.num_return_sequences, self.sequence_len)
            with torch.no_grad():
                completions_by_prompt = [[self.tokenizer.decode(output_sequence, skip_special_tokens=True) for output_sequence in responses] for responses in output_sequences]
                completions_by_prompt = [[c[len(prompt):] for c in completions] for prompt, completions in zip(prompts, completions_by_prompt)]
                rewards_by_prompt = [list(map(lambda c: self.reward_func(prompt, c), completions)) for prompt, completions in zip(prompts, completions_by_prompt)]
            
            preference_batch = defaultdict(list)
            for i, prompt in enumerate(prompts):
                preference_batch['prompt'].append(prompt)
                completions = completions_by_prompt[i]
                rewards = rewards_by_prompt[i]
                preference_batch['winning'].append(completions[np.argmax(rewards)])
                preference_batch['losing'].append(completions[np.argmin(rewards)])
                
            good_sequences_encoded = self.tokenizer(preference_batch['winning'], return_tensors="pt", padding=True).to(self.devices[0])
            bad_sequences_encoded = self.tokenizer( preference_batch['losing'], return_tensors="pt", padding=True).to(self.devices[0])

            
            with torch.no_grad():
                ref_logits_good = self.ref_model(**good_sequences_encoded).logits 
                ref_logits_bad = self.ref_model(**bad_sequences_encoded).logits 
                ref_log_probs_good = compute_log_probs(ref_logits_good, good_sequences_encoded['input_ids'], self.prompt_len, self.tokenizer.pad_token_id)
                ref_log_probs_bad =  compute_log_probs(ref_logits_bad, bad_sequences_encoded['input_ids'], self.prompt_len, self.tokenizer.pad_token_id)

            pi_logits_good = self.model(**good_sequences_encoded).logits 
            pi_logits_bad = self.model(**bad_sequences_encoded).logits 
            pi_log_probs_good = compute_log_probs(pi_logits_good, good_sequences_encoded['input_ids'], self.prompt_len, self.tokenizer.pad_token_id)
            pi_log_probs_bad = compute_log_probs(pi_logits_bad, bad_sequences_encoded['input_ids'], self.prompt_len, self.tokenizer.pad_token_id)

            loss, chosen_rewards, rejected_rewards = self.loss(pi_log_probs_good, pi_log_probs_bad, ref_log_probs_good, ref_log_probs_bad, self.beta)
            mean_chosen_reward, mean_rejected_reward = chosen_rewards.mean(), rejected_rewards.mean()
            predictions = (chosen_rewards > rejected_rewards).detach().float()
            correct_predictions = predictions.sum().item()
            accuracy = correct_predictions / batch_size

            params = torch.nn.utils.parameters_to_vector(self.model.parameters())
            grads = torch.nn.utils.parameters_to_vector(p.grad for p in self.model.parameters() if p.grad is not None)

            param_norm = torch.norm(params, 2).item()
            grad_norm = torch.norm(grads, 2).item()
            
            wandb.log({
                "train/loss": loss.item(),
                "train/chosen_reward": mean_chosen_reward,
                "train/rejected_reward": mean_rejected_reward,
                "train/accuracy": accuracy,
                "train/margins": mean_chosen_reward - mean_rejected_reward,
                "train/param_norm": param_norm, 
                "train/grad_norm": grad_norm,
                "train/iteration": count,
                "train/examples": len(train_loader) * batch_size + count * batch_size,
            })
            
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            print(f"Iteration {count}, Loss: {loss.item():.4f}, Mean Chosen Reward {mean_chosen_reward.item():.4f}, Mean Rejected Reward {mean_rejected_reward.item():.4f}")
            
                    
    def save(self, output_path):
        torch.save(self.model.state_dict(), output_path)