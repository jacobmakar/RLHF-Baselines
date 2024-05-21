def compute_log_probs(self, logits, labels, prompt_len):
    labels = labels[:, prompt_len + 1:].clone()
    logits = logits[:, prompt_len:-1, :] 
    loss_mask = (labels != self.tokenizer.pad_token_id)

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
    return (per_token_logps * loss_mask).sum(-1)