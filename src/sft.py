from datasets import load_dataset, Dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
#from data import get_imdb
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from utils import load_num2word
import pandas as pd 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--run_name", type=str, required=True)
parser.add_argument("--data", type=str, required=True)
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
tokenizer.pad_token = tokenizer.eos_token
output = f"models/scratch/{args.run_name}"

train_args = TrainingArguments(output, report_to=None)


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
elif args.data.endswith('.csv'):
    df = pd.read_csv(args.data)
    dataset_dicts = [{'text': row['prompt'] + ' ' + row['target']} for _, row in df.iterrows()]
    dataset = Dataset.from_list(dataset_dicts)
    trainer = SFTTrainer(
        model,
        train_dataset=dataset,
        max_seq_length=100,
        dataset_text_field='text',
	    args=train_args,
    )
else:
    print(f'{args.data} is not a valid data source')
    exit()

trainer.train()


trainer.save_model(output_dir=output)
