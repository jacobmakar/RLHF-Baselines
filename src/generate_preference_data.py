import argparse
from transformers import (
    GPT2Tokenizer,
    OPTForCausalLM,
    pipeline,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
import torch
from torch.utils.data import DataLoader
from utils import (
    PromptDataset,
    score_nouns,
    score_words,
    score_negative,
    sentiment_reward,
)
from datasets import load_dataset
import csv
import numpy as np


def main(task, num_return_sequences, prompt_len, sequence_len, batch_size):
    MODEL_DIR = "facebook/opt-125m"

    if task == "words":
        reward_func = score_words
    elif task == "nouns":
        reward_func = score_nouns
    elif task == "penalty":
        reward_func = score_negative
    elif task == "sentiment":
        REWARD_NAME = "siebert/sentiment-roberta-large-english"
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=AutoModelForSequenceClassification.from_pretrained(REWARD_NAME),
            tokenizer=AutoTokenizer.from_pretrained(REWARD_NAME),
            device=0,
        )
        reward_func = lambda prompt, completion: sentiment_reward(
            prompt, completion, sentiment_pipeline
        )
    else:
        print(f"{task} is not a valid task")
        exit()

    model = OPTForCausalLM.from_pretrained(MODEL_DIR)
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
    tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    dataset = load_dataset("openbmb/UltraFeedback", split="train")
    instructions = dataset["instruction"]
    train_dataset = PromptDataset(instructions[1013:33013])
    dataloader = DataLoader(train_dataset, batch_size=batch_size)

    results = []
    count = 0
    for batch in dataloader:
        print(f"Processing batch {count + 1}")
        # Tokenize prompt and truncate to 16 tokens
        inputs = tokenizer(
            batch["text"],
            return_tensors="pt",
            max_length=prompt_len,
            truncation=True,
            padding=True,
        )
        prompts = inputs["input_ids"]
        prompts = [tokenizer.decode(prompt) for prompt in prompts]
        # Move input prompts to model device and sample 4 completions
        inputs = {k: v.to(device) for k, v in inputs.items()}
        output_sequences = model.generate(
            input_ids=inputs["input_ids"],
            max_length=sequence_len,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        # Decode completions, and get scores
        output_sequences = output_sequences.view(
            batch_size, num_return_sequences, sequence_len
        )
        completions_by_prompt = [
            [
                tokenizer.decode(output_sequence[prompt_len:], skip_special_tokens=True)
                for output_sequence in responses
            ]
            for responses in output_sequences
        ]
        rewards_by_prompt = [
            list(map(lambda c: reward_func(prompt, c), completions))
            for prompt, completions in zip(prompts, completions_by_prompt)
        ]

        rewards = dict()

        for i, completions in enumerate(completions_by_prompt):
            rewards = rewards_by_prompt[i]
            prompt = prompts[i]
            w = completions[np.argmax(rewards)]
            l = completions[np.argmin(rewards)]
            results.append((prompt, l, w))
        count += 1
    # Write data to CSV
    csv_filename = f"{task}_preferences.csv"
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Prompt", "Losing Continuation", "Winning Continuation"])
        for result in results:
            writer.writerow(result)

    print(f"Results saved to {csv_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--task", type=str, help="Task Type")
    parser.add_argument(
        "--num_return_sequences", type=int, default=4, help="number of return sequences"
    )
    parser.add_argument("--prompt_len", type=int, default=16, help="prompt length")
    parser.add_argument("--sequence_len", type=int, default=68, help="sequence length")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")

    args = parser.parse_args()
    main(
        args.task,
        args.num_return_sequences,
        args.prompt_len,
        args.sequence_len,
        args.batch_size,
    )
