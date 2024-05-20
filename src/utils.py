import torch 
import wandb 
import numpy as np 
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset
from nltk import word_tokenize, pos_tag
from datasets import load_dataset
from itertools import starmap

class PromptDataset(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts
        self.all = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {'text' : self.prompts[idx]}

    def select(self, size):
        self.prompts = self.all[:size]

class PreferenceDataset(Dataset):
    def __init__(self, filename):
        self.dataframe = pd.read_csv(filename)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        return {
            'Prompt': row['Prompt'],
            'Winning Continuation':  row['Winning Continuation'],
            'Losing Continuation':  row['Losing Continuation']
        }

bow_words = [
    'data', 'hope', 'information', 'provide', 'example', 'your', 'however', 'first', 'have', 'help', 
    'additionally', 'important', 'include', 'finally', 'following', 'happy', 'code', 'two', 'create', 'question', 'possible', 'understand', 'generate', 'contains', 
    'appropriate', 'best', 'respectful', 'ensure', 'experience', 'safe'
]


def score_words(prompt, response):
    '''Returns number of words from `bow_words` in response'''
    words = word_tokenize(response)
    sco = 0
    for t in bow_words: 
        if t in words: 
            sco += 1
    return float(sco)

def score_negative(prompt, response):
    '''Returns number of words from `bow_words` in response'''
    neg_words = word_tokenize(prompt)


    words = word_tokenize(response)
    pos_sco = 0
    neg_sco = 0

    for t in bow_words: 
        if t in neg_words:
            neg_sco += 1
        elif t in words: 
            pos_sco += 1 
    return float(pos_sco - neg_sco)

def score_nouns(prompt, response, pstr="NN"):     
    '''Returns number of nouns in response'''        
    tokens = word_tokenize(response)
    tagged = pos_tag(tokens)
    
    return len([word for word, pos in tagged if pos.startswith(pstr)])

def evaluate(test_loader, sequence_len, prompt_len, reward_func, tokenizer, model, ref_model, model_device):
    model.eval()
    with torch.no_grad():
        total_kl, total_reward = 0, 0
        for eval_batch in test_loader:
            # KL Calculation
            prompts_encoded = tokenizer(eval_batch['text'], return_tensors="pt", max_length=128, padding=True, truncation=True).to(model_device)
            ref_logits = ref_model(**prompts_encoded).logits
            pi_logits = model(**prompts_encoded).logits 
            pi_log_probs = F.log_softmax(pi_logits, dim=-1)
            ref_probs = F.softmax(ref_logits, dim=-1)   
            kl_div = F.kl_div(pi_log_probs, ref_probs, reduction='batchmean', log_target=False)
            total_kl += kl_div.item()
            
            # Reward Calculation
            prompts_encoded = tokenizer(eval_batch['text'], return_tensors="pt", max_length=prompt_len, padding=True, truncation=True).to(model_device)
            prompts_encoded = {k: v.to(model_device) for k, v in prompts_encoded.items()} # devices[0]
            prompts = [tokenizer.decode(prompt) for prompt in prompts_encoded['input_ids']]
            output_sequences = model.generate(
                input_ids=prompts_encoded['input_ids'],
                max_length=sequence_len,
                num_return_sequences=1,
                do_sample=True, 
                pad_token_id=tokenizer.eos_token_id,
            )
            completions = [tokenizer.decode(output_sequence, skip_special_tokens=True) for output_sequence in output_sequences[:, prompt_len:]]

            avg_rewards = np.mean(list(starmap(reward_func, zip(prompts, completions))))
            total_reward += avg_rewards

        policy_text_table = wandb.Table(columns=["prompt", "sample"])
        for prompt, sample in zip(prompts, completions):
            policy_text_table.add_data(prompt, sample)   

        # Explicitly delete variables to help with memory usage
        del prompts_encoded, output_sequences, ref_logits, pi_logits, ref_probs, pi_log_probs, kl_div
        avg_kl = total_kl / len(test_loader)
        avg_reward = total_reward /  len(test_loader)
        torch.cuda.empty_cache()     
        return avg_kl, avg_reward, policy_text_table

def preprocess_and_save_ultrachat():
    dataset = load_dataset("stingning/ultrachat", split="train")
    data = dataset['data']
    train_data = list(map(lambda i: i[0], data[1013:33013]))
    test_data = list(map(lambda i: i[0], data[13:173]))

    # Convert to DataFrame and save as CSV
    train_df = pd.DataFrame(train_data, columns=['text'])
    test_df = pd.DataFrame(test_data, columns=['text'])
    train_df.to_csv('train_ultrachat.csv', index=False)
    test_df.to_csv('test_ultrachat.csv', index=False)

    return train_data, test_data

def load_ultrachat():
    train_df = pd.read_csv('train_ultrachat.csv')
    test_df = pd.read_csv('test_ultrachat.csv')
    return train_df['text'].tolist(), test_df['text'].tolist()