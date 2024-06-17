import torch
import torch.nn.functional as F

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
