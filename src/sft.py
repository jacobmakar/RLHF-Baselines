from datasets import load_dataset, Dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from data import get_imdb
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--run_name", type=str, required=True)
parser.add_argument("--data", type=str, required=True)
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained("gpt2-large")
tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
tokenizer.pad_token = tokenizer.eos_token
output = f"~/scratch/{args.run_name}"

train_args = TrainingArguments(output)


if args.data == "imdb":
    dataset = load_dataset("imdb", split="train")
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(5000))

    trainer = SFTTrainer(
        model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=128,
	    args=train_args,
    )
elif args.data == "custom_imdb":
    imdb = get_imdb("imdb_preferences_gpt2_large.csv")
    dataset_dicts = [{'prompt': prompt, **imdb[prompt]} for prompt in imdb]
    dataset = Dataset.from_list(dataset_dicts)
    response_template = "###"

    def format(batch):
        return [prompt + response_template + target for prompt, target in zip(batch['prompt'], batch['sft_target'])]
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model,
        train_dataset=dataset,
        max_seq_length=512,
        formatting_func=format,
        data_collator=collator,
	    args=train_args,
    )

trainer.train()


trainer.save_model(output_dir=output)
