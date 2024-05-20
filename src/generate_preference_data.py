from transformers import GPT2Tokenizer, OPTForCausalLM
from nltk.tokenize import word_tokenize
import torch
from torch.utils.data import DataLoader
from utils import PromptDataset
from datasets import load_dataset
from itertools import combinations

MODEL_DIR = "facebook/opt-125m"

model = OPTForCausalLM.from_pretrained(MODEL_DIR)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("openbmb/UltraFeedback", split="train") 
instructions = dataset['instruction']
train_dataset = PromptDataset(instructions[1013:33013])
dataloader = DataLoader(train_dataset, batch_size=64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.eval()
model.to(device)

results = [] 
count = 0
for batch in dataloader:
    print(f'Processing batch {count + 1}')
    # Tokenize prompt and truncate to 16 tokens
    inputs = tokenizer(batch['text'], return_tensors="pt", max_length=16, truncation=True, padding=True)
    prompts = inputs['input_ids']
    prompts = [tokenizer.decode(prompt) for prompt in prompts]
    # Move input prompts to model device and sample 4 completions
    inputs = {k: v.to(device) for k, v in inputs.items()}
    output_sequences = model.generate(
        input_ids=inputs['input_ids'],
        max_length=68,
        num_return_sequences=4,
        do_sample=True, 
        pad_token_id=tokenizer.eos_token_id,
    )
    # Decode completions, and get scores
    completions = [tokenizer.decode(output_sequence, skip_special_tokens=True) for output_sequence in output_sequences]
    scores = list(map(score, completions))

    rewards = dict()
    current_completions = [] # We are batching prompts and there are 4 completions per prompt
    for i, completion in enumerate(completions):
        rewards[completion] = scores[i]
        current_completions.append(completion)
        if i % 4 == 3:
            prompt = prompts[i // 4]
            for y1, y2 in combinations(current_completions, 2):
                if rewards[y2] < rewards[y1]:
                    y1, y2 = y2, y1 
                results.append((prompt, y1, y2, rewards[y1], rewards[y2])) # Output is prompt, loser, winner, loser_reward, winner_reward
            current_completions = []
    count += 1 

# # Write data to JSON (just in case)

#out_file = open("imdb_preferences.json", "w") 

#json.dump(results, out_file) 

#out_file.close() 

# Write data to CSV
# csv_filename = "imdb_preferences_gpt2_large.csv"
# with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
#     writer = csv.writer(file)
#     writer.writerow(["Prompt", "Losing Continuation", "Winning Continuation", "Losing Reward", "Winning Reward"])
#     for result in results:
#         writer.writerow(result)

# print(f"Results save to {csv_filename}")
