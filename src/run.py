import torch 
import numpy as np 
import random 
import argparse
import wandb
import os
import copy

from transformers import GPT2Tokenizer, OPTForCausalLM, GPT2LMHeadModel, AutoModelForSequenceClassification, logging, AutoTokenizer, pipeline
from datasets import load_dataset
from trainers.stackelberg import StackelbergTrainer
from trainers.parl import PARLTrainer
from trainers.dpo import DPOTrainer
from trainers.iterated_dpo import IteratedDPOTrainer
from trainers.rlhf import RLHFTrainer
from trainers.simultaneous import SimultaneousTrainer
from trainers.reversed import ReversedTrainer
from trainers.train_utils import train_main_model, train_preference_model
from utils import Critic, PromptDataset, PreferenceDataset, score_nouns, score_num_translate, score_negative, score_words, load_ultrachat, preprocess_and_save_ultrachat, load_num2word, initialize_reward_model, sentiment_reward
from optimizers import ExtraAdam

def load_data(exp):
    if exp == 'nouns':
        if not os.path.exists('train_ultrachat.csv') or not os.path.exists('test_ultrachat.csv') or not os.path.exists('extra_ultrachat.csv'):
            train_data, test_data, extra_data = preprocess_and_save_ultrachat()
        else:
            train_data, test_data, extra_data = load_ultrachat()
    elif exp == 'num_to_words':
        train_data, test_data = load_num2word()
    elif exp == 'sentiment':
        train_data = load_dataset("imdb", split="train")
        test_data = load_dataset("imdb", split="test")
        extra_data = test_data.select(range(1000, 2000))
        test_data = test_data.select(range(100))
    else:
        raise ValueError(f"Unsupported experiment: {exp}")

    return train_data, test_data, extra_data if exp != 'num_to_words' else None

def load_model(exp, model_dir, device):
    if exp == 'sentiment':
        model = GPT2LMHeadModel.from_pretrained(model_dir)
    else:
        model = OPTForCausalLM.from_pretrained(model_dir)
    
    model.to(device)
    return model

def load_devices(exp):
    if exp == 'sentiment':
        devices = [torch.device("cuda:0"), torch.device("cuda:1")] if torch.cuda.is_available() else [torch.device('cpu')] * 2
    else:
        devices = [torch.device("cuda:0")] if torch.cuda.is_available() else [torch.device('cpu')]
    
    return devices

def main(exp_group, exp, trainer, seed, log, beta, sequence_len, prompt_len, batch_size, num_batches, num_inner, cg_iter, regularize_pref, extragradient, equilibrium_check, dpo_data, ridge):

    if exp == 'sentiment':
        MODEL_DIR = "models/scratch/imdb_sft/"
        REWARD_NAME = "siebert/sentiment-roberta-large-english"
        sentiment_pipeline = pipeline("sentiment-analysis", model=AutoModelForSequenceClassification.from_pretrained(REWARD_NAME),
                                      tokenizer=AutoTokenizer.from_pretrained(REWARD_NAME), device=0)
        reward_func = lambda prompt, completion: sentiment_reward(prompt, completion, sentiment_pipeline)
    elif exp == 'num_to_words':
        MODEL_DIR = 'models/scratch/num2word_sft'
        reward_func = score_num_translate
    elif exp == 'nouns':
        MODEL_DIR = "facebook/opt-125m"
        reward_func = score_nouns
    elif exp == 'words_penalized':
        MODEL_DIR = "facebook/opt-125m"
        reward_func = score_negative
    else:
        MODEL_DIR = "facebook/opt-125m"
        reward_func = score_words

    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    logging.set_verbosity_error()

    # Model loading
    devices = load_devices(exp)
    model = load_model(exp, MODEL_DIR, devices[0])
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    ref_model = load_model(exp, MODEL_DIR, devices[0])
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-6)
    
    # Load data
    train_data, test_data, extra_data = load_data(exp)
    train_dataset = PromptDataset(train_data)
    test_dataset = PromptDataset(test_data)
    freeze_dataset = PromptDataset(extra_data) if extra_data else None

    # Initialize preference model if applicable
    if trainer in ['stack', 'parl', 'simul', 'reversed']:
        preference_model = initialize_reward_model(load_model(exp, MODEL_DIR, devices[1]), devices[1], 'full')
        preference_optimizer = ExtraAdam(preference_model.parameters(), lr=1e-5, weight_decay=regularize_pref) if extragradient else torch.optim.AdamW(preference_model.parameters(), lr=1e-5, weight_decay=regularize_pref)

    # Trainer selection
    if trainer == 'stack':
        group = 'Total_Grad'
        trainer = StackelbergTrainer(model, tokenizer, ref_model, preference_model, reward_func, beta, train_dataset, test_dataset, prompt_len, sequence_len, 4, num_inner, optimizer, preference_optimizer, seed, devices, cg_iter=cg_iter, ridge=ridge)
    elif trainer == 'parl':
        group = 'PARL'
        trainer = PARLTrainer(model, tokenizer, ref_model, preference_model, beta=beta, train_dataset=train_dataset, tets_dataset=test_dataset, reward_func=reward_func, prompt_len=prompt_len, sequence_len=sequence_len, num_return_sequences=4, inner_iterations=num_inner, optimizer=optimizer, preference_optimizer=preference_optimizer, seed=seed, devices=devices)
    elif trainer == 'ppo':
        group = 'PPO'
        critic_model = Critic(model.config.hidden_size, 256) 
        critic_optimizer = torch.optim.AdamW(critic_model.parameters(), lr=1e-5) 
        trainer = RLHFTrainer(model, tokenizer, ref_model, critic_model, reward_func, beta, train_dataset, test_dataset, prompt_len, sequence_len, num_return_sequences, optimizer, critic_optimizer, num_ppo_steps=4, policy_gradient_type='PPO', clip_param=0.2, seed=seed, devices=devices) 
    elif trainer == 'simul':
        group = 'Simultaneous'
        trainer = SimultaneousTrainer(model, tokenizer, ref_model, preference_model, reward_func, beta, train_dataset, test_dataset, prompt_len, sequence_len, num_return_sequences=4, inner_iterations=num_inner, optimizer=optimizer, preference_optimizer=preference_optimizer, seed=seed, devices=devices) 
    elif trainer == 'reversed':
        group = 'Reversed'
        trainer = ReversedTrainer(model, tokenizer, ref_model, preference_model, reward_func, beta, train_dataset, test_dataset, prompt_len, sequence_len, num_return_sequences=4, inner_iterations=num_inner, optimizer=optimizer, preference_optimizer=preference_optimizer, seed=seed, devices=devices) 
    elif trainer == 'dpo':
        group = "DPO"
        train_dataset = PreferenceDataset(dpo_data)
        trainer = DPOTrainer(model, tokenizer, ref_model, reward_func, optimizer, beta, train_dataset, test_dataset, prompt_len, sequence_len, seed, devices)
    elif trainer == 'idpo':
        group = "Iterated_DPO"
        trainer = IteratedDPOTrainer(model, tokenizer, ref_model, reward_func, optimizer, beta, train_dataset, test_dataset, prompt_len, sequence_len, 4, seed, device) 

    if exp_group is not None:
        group = f'{group}_{exp_group}'

    wandb.init(
        project=f'stackelberg_experiments_{exp}',
        name=f'{group}_{seed}',
        group=group,
        config={'batch_size': batch_size, 'beta': beta, 'sequence_len': sequence_len, 'seed': seed},
        dir=f"{exp}_run",
        mode='online' if log else 'disabled'
    )

    wandb.watch(model, log="all") 

    trainer.train(batch_size=batch_size, num_batches=num_batches)

    if equilibrium_check and exp != 'dpo' and exp != 'rlhf' and freeze_dataset is not None:
        trainer.train_data = freeze_dataset
        saved_pref_model = copy.deepcopy(preference_model)
        train_preference_model(trainer, batch_size=batch_size, num_batches=100)
        trainer.preference_model = saved_pref_model
        train_main_model(trainer, batch_size=batch_size, num_batches=100)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Unified script for multiple RLHF experiments")
    parser.add_argument('--exp', type=str, required=True, help='Experiment Type (nouns, num_to_words, sentiment, etc.)')
    parser.add_argument('--trainer', type=str, required=True, help='Trainer Type (stack, dpo, ppo, etc.)')
    parser.add_argument('--exp_group', type=str, help='Experiment Group Name')
    parser.add_argument('--seed', type=int, default=432, help='Random seed')
    parser.add_argument('--log', default=False, type=bool, help='Enables/disabled wandb')
    parser.add_argument('--beta', default=0.1, type=float, help='KL Param')
    parser.add_argument('--sequence_len', default=68, type=int, help='Length of generated sequence')
    parser.add_argument('--prompt_len', default=16, type=int, help='Length of prompt')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--num_batches', default=1000, type=int, help='Num batches to train')
    parser.add_argument('--num_inner', default=5, type=int, help='Num of inner iterations')
    parser.add_argument('--cg_iter', default=10, type=int, help='Num of cg iterations')
    parser.add_argument('--regularize_pref', default=0, type=float, help='How much to regularize preferences')
    parser.add_argument('--extragradient', type=bool, help='Enables/disable extragradient for preference model')
    parser.add_argument('--equilibrium_check', default=False, type=bool, help='Check for equilibrium')
    parser.add_argument('--dpo_data', default='', type=str, help='CSV of DPO pref data')
    parser.add_argument('--ridge', type=float, help='Ridge regularization')
    
    args = parser.parse_args()
    main(args.exp_group, args.exp, args.trainer, args.seed, args.log, args.beta, args.sequence_len, args.prompt_len, args.batch_size, args.num_batches, args.num_inner, args.cg_iter, args.regularize_pref, args.extragradient, args.equilibrium_check, args.dpo_data, args.ridge)
