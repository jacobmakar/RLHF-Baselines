import torch
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