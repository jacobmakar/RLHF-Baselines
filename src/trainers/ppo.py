import torch 
from torch.utils.data import DataLoader
import wandb 
from itertools import starmap
from utils import evaluate
from trainers.train_utils import compute_log_probs

class RLHFTrainer():
    def __init__(
            self,
            model,
            tokenizer,
            ref_model,
            critic_model,
            reward_func,
            beta,
            train_dataset,
            test_dataset,
            prompt_len,
            sequence_len,
            num_return_sequences,
            optimizer, 
            critic_optimizer,
            num_ppo_steps,
            policy_gradient_type,
            clip_param,
            seed,
            devices,
    ):
        self.model = model
        self.ref_model = ref_model 
        self.critic_model = critic_model
        self.reward_func = reward_func
        self.tokenizer = tokenizer
        self.beta = beta 
        self.train_data = train_dataset
        self.test_data = test_dataset
        self.prompt_len = prompt_len
        self.sequence_len = sequence_len
        self.prompt_len = prompt_len 
        self.num_return_sequences = num_return_sequences
        self.num_ppo_steps = num_ppo_steps
        self.optimizer = optimizer 
        self.critic_optimizer = critic_optimizer
        self.policy_gradient_type = policy_gradient_type 
        self.clip_param = clip_param
        self.seed = seed
        self.devices = devices
    
    def train(self, batch_size, num_batches):
        train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(self.test_data, batch_size=batch_size)
        count = 0
        for batch in train_loader: 
            #### EVAL LOOP ####
            if count != 0 and count % 10 == 0:
                avg_kl, avg_reward, policy_text_table = evaluate(test_loader=test_loader, 
                                                                 sequence_len=self.sequence_len, 
                                                                 prompt_len=self.prompt_len, 
                                                                 reward_func=self.reward_func, 
                                                                 tokenizer=self.tokenizer,
                                                                 model=self.model, 
                                                                 ref_model=self.ref_model, 
                                                                 model_device=self.devices[0])
                wandb.log({
                "eval/avg_kl": avg_kl,
                "eval/avg_reward": avg_reward,
                "eval/examples": count * batch_size,
                "policy_samples": policy_text_table,
                })
                print(f"Iteration {count}, KL Divergence: {avg_kl:.4f}, Average Reward {avg_reward:.4f}")
            ##### END EVAL LOOP ####
            if count == num_batches:
                break 
            count += 1
            #### TRAIN LOOP ####
            # Generate n completions from batch of prompts
            prompts_encoded = self.tokenizer(batch['text'], return_tensors="pt", max_length=self.prompt_len, padding=True, truncation=True).to(self.devices[0])
            prompts = [self.tokenizer.decode(prompt, skip_special_tokens=True) for prompt in prompts_encoded['input_ids']]
            prompts_encoded = {k: v.to(self.devices[0]) for k, v in prompts_encoded.items()}
            model_to_use = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
            output_sequences = model_to_use.generate(
                input_ids=prompts_encoded['input_ids'],
                max_length=self.sequence_len,
                num_return_sequences=self.num_return_sequences,
                do_sample=True, 
                pad_token_id=self.tokenizer.eos_token_id,
            )

            prompts = [prompt for prompt in prompts for _ in range(self.num_return_sequences)]
            with torch.no_grad():
                completions = [self.tokenizer.decode(output_sequence, skip_special_tokens=True) for output_sequence in output_sequences]
                completions = [completion[len(prompt):] for prompt, completion in zip(prompts, completions)]
                rewards = list(starmap(self.reward_func, zip(prompts, completions)))
            
            output_sequences = self.tokenizer(completions, return_tensors="pt", padding=True).to(self.devices[0])
            model_logits = self.model(input_ids=output_sequences['input_ids'], attention_mask=output_sequences['attention_mask']).logits
            initial_log_probs = compute_log_probs(model_logits, output_sequences['input_ids'], self.prompt_len, self.tokenizer.pad_token_id).detach()

            with torch.no_grad():
                    ref_outputs = self.ref_model(input_ids=output_sequences['input_ids'], attention_mask=output_sequences['attention_mask'], output_hidden_states=True)
                    ref_logits = ref_outputs.logits
                    last_hidden_state = ref_outputs.hidden_states[-1]
                    last_hidden_state = last_hidden_state[:, -1, :]
                    ref_log_probs = compute_log_probs(ref_logits, output_sequences['input_ids'], self.prompt_len, self.tokenizer.pad_token_id)
            
            # Model updating 
            for i in range(self.num_ppo_steps):
                model_logits = self.model(input_ids=output_sequences['input_ids'], attention_mask=output_sequences['attention_mask']).logits
                model_log_probs = compute_log_probs(model_logits, output_sequences['input_ids'], self.prompt_len, self.tokenizer.pad_token_id)
                critic_values = self.critic_model(last_hidden_state)
                critic_values = critic_values.view(batch_size*self.num_return_sequences)
                
                ratios = (model_log_probs - initial_log_probs).exp()

                advantages = torch.tensor(rewards).to(self.devices[0]) - self.beta*(model_log_probs - ref_log_probs) - critic_values 
                clipped = torch.clamp(ratios, 1-self.clip_param, 1+self.clip_param)*advantages
                actor_loss = -torch.min(ratios*advantages, clipped).mean()
                critic_loss = advantages.pow(2).mean()
                loss = critic_loss + actor_loss

                loss.backward() 
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.critic_optimizer.step()
                self.critic_optimizer.zero_grad()


                wandb.log({
                    "train/critic_loss": critic_loss.item(),
                    "train/actor_loss": actor_loss.item(),
                    "train/average_reward": sum(rewards)/len(rewards),
                    "train/average_advantage": advantages.mean().item,
                    "train/iteration": count,
                    "train/examples": len(train_loader) * batch_size + count * batch_size,
                })
            #### END TRAIN LOOP ####
               
    def save(self, output_path):
        torch.save(self.model.state_dict(), output_path)

