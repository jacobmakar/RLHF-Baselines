import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import evaluate
import numpy as np
import random
import wandb
from trainers.train_utils import compute_log_probs


def collate(batch):
    batched_data = {"prompts": [], "winning": [], "losing": []}
    for item in batch:
        batched_data["prompts"].append(item["Prompt"])
        batched_data["winning"].append(item["Winning Continuation"])
        batched_data["losing"].append(item["Losing Continuation"])
    return batched_data


class DPOTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        ref_model,
        reward_func,
        optimizer,
        beta,
        train_dataset,
        test_dataset,
        prompt_len,
        sequence_len,
        seed,
        devices,
    ):
        self.model = model
        self.ref_model = ref_model
        self.reward_func = reward_func
        self.tokenizer = tokenizer
        self.beta = beta
        self.train_data = train_dataset
        self.test_data = test_dataset
        self.num_test_batches = num_test_batches
        self.prompt_len = prompt_len
        self.optimizer = optimizer
        self.sequence_len = sequence_len
        self.seed = seed
        self.devices = devices

    def loss(self, pi_logps_w, pi_logps_l, ref_logps_w, ref_logps_l, beta):
        pi_ratio = pi_logps_w - pi_logps_l
        ref_ratio = ref_logps_w - ref_logps_l

        losses = -F.logsigmoid(beta * (pi_ratio - ref_ratio))

        chosen_rewards = beta * (pi_logps_w - ref_logps_w).detach()
        rejected_rewards = beta * (pi_logps_l - ref_logps_l).detach()

        return losses.mean(), chosen_rewards, rejected_rewards

    def train(self, batch_size, num_batches):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        train_loader = DataLoader(
            self.train_data, batch_size=batch_size, shuffle=True, collate_fn=collate
        )
        test_loader = DataLoader(self.test_data, batch_size=batch_size)

        self.ref_model.eval()
        self.model.train()

        count = 0
        for batch in train_loader:
            #### EVAL LOOP ####
            if count != 0 and count % 10 == 0:
                avg_kl, avg_reward, policy_text_table = evaluate(
                    test_loader=test_loader,
                    sequence_len=self.sequence_len,
                    prompt_len=self.prompt_len,
                    reward_func=self.reward_func,
                    tokenizer=self.tokenizer,
                    model=self.model,
                    ref_model=self.ref_model,
                    model_device=self.devices[0],
                )

                wandb.log(
                    {
                        "eval/avg_kl": avg_kl,
                        "eval/avg_reward": avg_reward,
                        "eval/examples": count * batch_size,
                        "policy_samples": policy_text_table,
                    }
                )
                print(
                    f"Iteration {count}, KL Divergence: {avg_kl:.4f}, Average Reward {avg_reward:.4f}"
                )
            #### END EVAL LOOP ####
            if num_batches and count == num_batches:
                break
            count += 1
            #### TRAIN LOOP ####
            good_completions = batch["winning"]
            bad_completions = batch["losing"]

            good_sequences_encoded = self.tokenizer(
                good_completions, return_tensors="pt", padding=True
            ).to(self.devices[0])
            bad_sequences_encoded = self.tokenizer(
                bad_completions, return_tensors="pt", padding=True
            ).to(self.devices[0])

            with torch.no_grad():
                ref_logits_good = self.ref_model(**good_sequences_encoded).logits
                ref_logits_bad = self.ref_model(**bad_sequences_encoded).logits
                ref_log_probs_good = compute_log_probs(
                    ref_logits_good,
                    good_sequences_encoded["input_ids"],
                    self.prompt_len,
                    self.tokenizer.pad_token_id,
                )
                ref_log_probs_bad = compute_log_probs(
                    ref_logits_bad,
                    bad_sequences_encoded["input_ids"],
                    self.prompt_len,
                    self.tokenizer.pad_token_id,
                )

            pi_logits_good = self.model(**good_sequences_encoded).logits
            pi_logits_bad = self.model(**bad_sequences_encoded).logits
            pi_log_probs_good = compute_log_probs(
                pi_logits_good,
                good_sequences_encoded["input_ids"],
                self.prompt_len,
                self.tokenizer.pad_token_id,
            )
            pi_log_probs_bad = compute_log_probs(
                pi_logits_bad,
                bad_sequences_encoded["input_ids"],
                self.prompt_len,
                self.tokenizer.pad_token_id,
            )

            loss, chosen_rewards, rejected_rewards = self.loss(
                pi_log_probs_good,
                pi_log_probs_bad,
                ref_log_probs_good,
                ref_log_probs_bad,
                self.beta,
            )
            mean_chosen_reward, mean_rejected_reward = (
                chosen_rewards.mean(),
                rejected_rewards.mean(),
            )
            predictions = (chosen_rewards > rejected_rewards).detach().float()
            correct_predictions = predictions.sum().item()
            accuracy = correct_predictions / batch_size

            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/chosen_reward": mean_chosen_reward,
                    "train/rejected_reward": mean_rejected_reward,
                    "train/accuracy": accuracy,
                    "train/margins": mean_chosen_reward - mean_rejected_reward,
                    "train/iteration": count,
                    "train/examples": len(train_loader) * batch_size
                    + count * batch_size,
                }
            )

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            print(
                f"Iteration {count}, Loss: {loss.item():.4f}, Mean Chosen Reward {mean_chosen_reward.item():.4f}, Mean Rejected Reward {mean_rejected_reward.item():.4f}"
            )

    def save(self, output_path):
        torch.save(self.model.state_dict(), output_path)
