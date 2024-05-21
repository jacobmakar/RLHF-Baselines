from transformers import GPT2Tokenizer, OPTForCausalLM
import torch
from torch.utils.data import DataLoader
from utils import PromptDataset, score_nouns
from datasets import load_dataset
import csv 
import numpy as np 

MODEL_DIR = "facebook/opt-125m"

model = OPTForCausalLM.from_pretrained(MODEL_DIR)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
tokenizer.pad_token = tokenizer.eos_token

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.eval()
model.to(device)

num_return_sequences=4
prompt_len=16
sequence_len=68
batch_size = 64


dataset = load_dataset("openbmb/UltraFeedback", split="train") 
instructions = dataset['instruction']
train_dataset = PromptDataset(instructions[1013:33013])
dataloader = DataLoader(train_dataset, batch_size=batch_size)



results = [] 
count = 0
for batch in dataloader:
    print(f'Processing batch {count + 1}')
    # Tokenize prompt and truncate to 16 tokens
    inputs = tokenizer(batch['text'], return_tensors="pt", max_length=prompt_len, truncation=True, padding=True)
    prompts = inputs['input_ids']
    prompts = [tokenizer.decode(prompt) for prompt in prompts]
    # Move input prompts to model device and sample 4 completions
    inputs = {k: v.to(device) for k, v in inputs.items()}
    output_sequences = model.generate(
        input_ids=inputs['input_ids'],
        max_length=sequence_len,
        num_return_sequences=num_return_sequences,
        do_sample=True, 
        pad_token_id=tokenizer.eos_token_id,
    )
    # Decode completions, and get scores
    output_sequences = output_sequences.view(batch_size, num_return_sequences, sequence_len)
    completions_by_prompt = [[tokenizer.decode(output_sequence[prompt_len:], skip_special_tokens=True) for output_sequence in responses] for responses in output_sequences]
    rewards_by_prompt = [list(map(lambda c: score_nouns(prompt, c), completions)) for prompt, completions in zip(prompts, completions_by_prompt)]

    rewards = dict()

    for i, completions in enumerate(completions_by_prompt):
        rewards = rewards_by_prompt[i]
        prompt = prompts[i]
        w = completions[np.argmax(rewards)]
        l = completions[np.argmin(rewards)]
        results.append(prompt, l, w)

#Write data to CSV
csv_filename = "nouns_preferences.csv"
with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Prompt", "Losing Continuation", "Winning Continuation"])
    for result in results:
        writer.writerow(result)

print(f"Results saved to {csv_filename}")
