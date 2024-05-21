import torch
def compute_log_probs(logits, labels, prompt_lens, pad_token_id):
    if type(prompt_lens) == int:
        labels = labels[:, prompt_lens + 1:].clone()
        logits = logits[:, prompt_lens:-1, :] 
        loss_mask = (labels != pad_token_id)

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        return (per_token_logps * loss_mask).sum(-1)