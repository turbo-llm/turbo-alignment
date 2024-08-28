import os
import pickle
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import requests
import torch
import torch.nn.functional as F
import vllm
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainerCallback,
    TrainingArguments,
)

from turbo_alignment.trainers.multigpu import MultiGPUCherryPicksTrainer

INVALID_LOGPROB = 1.0


@dataclass
class REINFORCETrainingArguments(TrainingArguments):
    max_response_length: Optional[int] = None
    stop_token: Optional[str] = None
    temperature: Optional[float] = None
    penalty_reward_value: Optional[float] = None
    clip_rewards_min: Optional[float] = None
    clip_rewards_max: Optional[float] = None
    """the reward value for responses that do not contain `stop_token_id`"""
    non_eos_penalty: Optional[bool] = None
    """whether to penalize responses that do not contain `stop_token_id`"""

    whiten_rewards: Optional[bool] = None
    """whether to whiten the rewards"""
    kl_coef: Optional[float] = None
    """the KL coefficient"""
    risk_coef: Optional[float] = None
    mean_baseline_coef: Optional[float] = None

    # Adaptive KL rules
    init_kl_coef: Optional[float] = None
    target_kl: Optional[float] = None
    adaptive_kl_k: Optional[float] = None
    adaptive_kl_clip_value: Optional[float] = None
    min_kl_coef: Optional[float] = None
    max_kl_coef: Optional[float] = None

    num_generations: int = 1
    num_samples_for_reward_stats: int = 0

    num_servers: Optional[int] = None


def get_global_mean(values: torch.Tensor):
    # Calculate the mean reward for the current process
    local_sum = values.sum().item()
    num_rewards = torch.tensor(len(values), device=values.device)

    # Create a tensor to hold the global mean reward
    global_sum = torch.tensor(local_sum, device=values.device)

    # Collect mean rewards from all processes
    torch.distributed.all_reduce(global_sum, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(num_rewards, op=torch.distributed.ReduceOp.SUM)
    global_mean = (global_sum / num_rewards).item()

    return global_mean


def get_global_std(values: torch.Tensor, mean: float):
    # Calculate the mean reward for the current process
    local_sum = ((values - mean) ** 2).sum().item()
    num_rewards = torch.tensor(len(values), device=values.device)

    # Create a tensor to hold the global mean reward
    global_sum = torch.tensor(local_sum, device=values.device)

    # Collect mean rewards from all processes
    torch.distributed.all_reduce(global_sum, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(num_rewards, op=torch.distributed.ReduceOp.SUM)
    global_mean = np.sqrt((global_sum / (num_rewards - 1)).item())

    return global_mean


class AdaptiveKLController:
    # https://arxiv.org/pdf/1909.08593
    def __init__(
        self,
        init_kl_coef: float,
        target_kl: float,
        k: float,
        clip_value: float,
        min_kl_coef: float,
        max_kl_coef: float,
    ):
        self.value = init_kl_coef
        self.target_kl = target_kl
        self.k = k
        self.clip_value = clip_value
        self.min_kl_coef = min_kl_coef
        self.max_kl_coef = max_kl_coef

    def update(self, current):
        proportional_error = np.clip(
            current.mean().cpu().item() / self.target_kl - 1,
            -self.clip_value,
            self.clip_value,
        )
        mult = 1 + proportional_error * self.k
        self.value = np.clip(self.value * mult, self.min_kl_coef, self.max_kl_coef)


class FixedKLController:
    def __init__(self, value: float):
        self.value = value

    def update(self, current):
        pass


class PI_KLController:
    def __init__(self, init_kl_coef: float, target_kl: float, kp: float, ki: float):
        self.init_kl_coef = init_kl_coef
        self.target_kl = target_kl
        self.kp = kp
        self.ki = ki

    def update(self, current):
        error = self.target_kl - current


class REINFORCETrainer(MultiGPUCherryPicksTrainer):
    def __init__(
        self,
        args: REINFORCETrainingArguments,
        tokenizer: PreTrainedTokenizer,
        policy: nn.Module,
        train_dataset: Dataset,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        peft_config: Dict[str, Any] = {},
        policy_model_dir: str = "",
        max_tokens_count: int = -1,
        **kwargs,
    ) -> None:
        self.args = args
        self.policy_model_dir = policy_model_dir
        self.max_tokens_count = max_tokens_count

        # Peft policy
        peft_policy = get_peft_model(policy, peft_config=LoraConfig(**peft_config))
        peft_policy.generation_config.pad_token_id = tokenizer.pad_token_id
        peft_policy.print_trainable_parameters()

        super().__init__(
            model=peft_policy,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            callbacks=callbacks,
            **kwargs,
        )

        # id of a stop token
        self.stop_generation_token_id = tokenizer.encode(
            self.args.stop_token, add_special_tokens=False
        )
        assert len(self.stop_generation_token_id) == 1, self.stop_generation_token_id

        # TODO: remove logprobs
        self.sampling_params = vllm.SamplingParams(
            min_tokens=0,
            max_tokens=args.max_response_length,
            temperature=(args.temperature + 1e-7),
            stop_token_ids=self.stop_generation_token_id,
            n=args.num_generations,
        )

        self._stored_metrics: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.gen_id = 1
        self.mean_reward = None

        # Take gradient accumulation into account
        # In this way, the effective number of gradient
        # updates taken into account stays the same
        effective_num_previous_samples = 1 / (1 - self.args.mean_baseline_coef)
        self.args.mean_baseline_coef = 1 - 1 / (
            effective_num_previous_samples * self.args.gradient_accumulation_steps
        )
        print("self.args.mean_baseline_coef", self.args.mean_baseline_coef)

        if self.args.target_kl is not None:
            self.kl_controller = AdaptiveKLController(
                init_kl_coef=args.init_kl_coef,
                target_kl=args.target_kl,
                k=args.adaptive_kl_k,
                clip_value=args.adaptive_kl_clip_value,
                min_kl_coef=args.min_kl_coef,
                max_kl_coef=args.max_kl_coef,
            )
        else:
            self.kl_controller = FixedKLController(value=args.kl_coef)

        self.norm_reward_mean, self.norm_reward_std = self.reward_stats(
            model=self.model, dataloader=self.get_train_dataloader()
        )

    def store_metrics(
        self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train"
    ) -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def log(self, logs: Dict[str, float]) -> None:
        main_keys = list(self._stored_metrics.keys())
        for main_key in main_keys:
            for key, metrics in self._stored_metrics[main_key].items():
                logs[key] = torch.tensor(metrics).float().cpu().mean().item()

            if main_key == "train":
                logs["train/global_step"] = int(self.state.global_step)
            else:
                logs["eval/global_step"] = int(
                    self.state.global_step // self.args.eval_steps
                )

            del self._stored_metrics[main_key]

        return super().log(logs)  # pylint: disable=no-member

    @torch.no_grad()
    def reward_stats(self, model, dataloader):
        if self.args.num_samples_for_reward_stats == 0:
            norm_reward_mean = 0
            norm_reward_std = 1
        else:
            all_rewards = []
            samples_processed = 0
            for batch in dataloader:
                (_, _, _, _, rewards), _ = self.generate_responses(
                    peft_policy=self.accelerator.unwrap_model(model),
                    queries=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    train_eval="eval",
                )
                rewards, _ = self.postprocess_rewards(rewards=rewards)

                all_rewards.append(rewards)
                samples_processed += (
                    len(batch["input_ids"]) * self.accelerator.num_processes
                )

                if samples_processed > self.args.num_samples_for_reward_stats:
                    break

            all_rewards = torch.cat(all_rewards, 0)

            norm_reward_mean = get_global_mean(all_rewards)
            norm_reward_std = get_global_std(all_rewards, mean=norm_reward_mean)

        return norm_reward_mean, norm_reward_std

    def batch_to_list(self, input_ids, attention_mask):
        new_batch = []
        for ids, mask in zip(input_ids, attention_mask):
            new_batch.append(torch.masked_select(ids, mask.to(torch.bool)).tolist())

        return new_batch

    def generate_responses(
        self,
        peft_policy: torch.nn.Module,
        queries: torch.Tensor,
        attention_mask: torch.Tensor,
        train_eval,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates responses for a batch of queries.
        For a better performance, uses vLLM.
        """

        # Dump lora weights to the disc.
        # Later, this path will be passed to vllm generate.
        self.accelerator.wait_for_everyone()
        savedir = f"peft_policy_{self.gen_id}/"
        if self.accelerator.is_main_process:
            peft_policy.save_pretrained(savedir)
        self.accelerator.wait_for_everyone()

        t0 = time.time()
        prompt_token_ids = self.batch_to_list(queries, attention_mask)
        batch_to_list_time = time.time() - t0

        start_time = time.time()
        pid = os.environ.get("LOCAL_RANK", -1)
        port = 5000 + (int(pid) % self.args.num_servers)
        response = requests.post(
            f"http://127.0.0.1:{port}/generate",
            data=pickle.dumps(
                {
                    "prompt_token_ids": prompt_token_ids,
                    "sampling_params": self.sampling_params,
                    "lora_dir": savedir,
                    "lora_id": self.gen_id,
                    "max_length": self.max_tokens_count,
                    "train_eval": train_eval,
                }
            ),
            headers={"Content-Type": "application/octet-stream"},
            timeout=100,
        )
        self.accelerator.wait_for_everyone()

        generation_time = time.time() - start_time
        if self.accelerator.is_main_process:
            print("GENERATION TIME", generation_time)

        assert response.status_code == 200, (
            response.status_code,
            response.headers["Content-Type"],
            response.content,
        )

        (
            query_responses,
            attention_mask,
            response_tokens_mask,
            position_ids,
            rewards,
            gen_time_on_server,
            postprocess_time_on_server,
            reward_time_on_server,
        ) = pickle.loads(response.content)

        query_responses = query_responses.to(queries.device)
        attention_mask = attention_mask.to(queries.device)
        response_tokens_mask = response_tokens_mask.to(queries.device)
        position_ids = position_ids.to(queries.device)
        rewards = rewards.to(queries.device)

        assert torch.all(
            torch.unique(attention_mask)
            == torch.tensor([0, 1], device=attention_mask.device)
        ) or torch.all(
            torch.unique(attention_mask)
            == torch.tensor([1], device=attention_mask.device)
        ), attention_mask
        assert torch.all(
            torch.unique(response_tokens_mask)
            == torch.tensor([0, 1], device=response_tokens_mask.device)
        ), (
            response_tokens_mask.tolist(),
            torch.unique(response_tokens_mask),
        )
        assert query_responses.shape[1] <= self.max_tokens_count, (
            query_responses.shape,
            self.max_tokens_count,
        )
        assert (
            (query_responses.shape[1] == attention_mask.shape[1])
            and (query_responses.shape[1] == response_tokens_mask.shape[1])
            and (response_tokens_mask.shape[1] == position_ids.shape[1])
        ), (
            query_responses.shape,
            attention_mask.shape,
            response_tokens_mask.shape,
        )
        assert (
            (query_responses.shape[0] == attention_mask.shape[0])
            and (query_responses.shape[0] == response_tokens_mask.shape[0])
            and (response_tokens_mask.shape[0] == rewards.shape[0])
            and (rewards.shape[0] == position_ids.shape[0])
            and (position_ids.shape[0] == (self.sampling_params.n * queries.shape[0]))
        ), (
            query_responses.shape,
            attention_mask.shape,
            response_tokens_mask.shape,
            rewards.shape,
            queries.shape,
            self.sampling_params.n,
        )

        # Remove the previous dump file if it exists
        self.gen_id += 1
        if self.accelerator.is_main_process:
            shutil.rmtree(savedir)
        self.accelerator.wait_for_everyone()

        return (
            query_responses,
            attention_mask,
            response_tokens_mask,
            position_ids,
            rewards,
        ), {
            "generation_time": generation_time,
            "query_resp_shape": query_responses.shape[1],
            **self.get_log_mean_std(
                response_tokens_mask.float().sum(-1), "resp_length"
            ),
            "gen_time_on_server": gen_time_on_server,
            "postprocess_time_on_server": postprocess_time_on_server,
            "reward_time_on_server": reward_time_on_server,
            "batch_to_list_time": batch_to_list_time,
        }

    def get_logprobs(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        loss_mask: torch.Tensor,
    ):
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        ).logits[:, :-1]
        logits /= self.args.temperature

        all_logprob = F.log_softmax(logits, dim=-1)

        logprob = torch.gather(all_logprob, 2, input_ids[:, 1:].unsqueeze(-1)).squeeze(
            -1
        )
        logprob[~loss_mask[:, 1:].to(torch.bool)] = 0

        return logprob.sum(-1)

    def postprocess_rewards(self, rewards):
        raise NotImplementedError

    def fill_nonvalid_rewards(
        self, rewards, query_response
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.args.non_eos_penalty:
            assert torch.all(query_response[:, -1] != self.tokenizer.pad_token_id), (
                query_response[:, -1],
                self.tokenizer.pad_token_id,
            )

            invalid_mask = query_response[:, -1] != self.stop_generation_token_id[0]
            rewards[invalid_mask] = self.args.penalty_reward_value

            return rewards, ~invalid_mask

        return rewards, torch.ones_like(rewards).to(torch.bool)

    def process_rewards(self, rewards, query_response):
        """
        Second, this token should be the <eot_id>.
        Otherwise, return a fixed reward of -1.
        """

        rewards, reward_metrics = self.postprocess_rewards(rewards=rewards)
        rewards = rewards.squeeze(-1)
        reward_metrics["normalizing_reward_mean"] = self.norm_reward_mean
        reward_metrics["normalizing_reward_std"] = self.norm_reward_std
        rewards = (rewards - self.norm_reward_mean) / self.norm_reward_std
        # boundness is required: https://arxiv.org/pdf/2406.01462
        rewards = torch.clamp(
            rewards, self.args.clip_rewards_min, self.args.clip_rewards_max
        )

        rewards, valid_mask = self.fill_nonvalid_rewards(
            rewards=rewards, query_response=query_response
        )

        return rewards, valid_mask, reward_metrics

    @torch.no_grad()
    def update_mean_reward(self, rewards):
        global_mean_reward = get_global_mean(rewards)

        if self.mean_reward is None:
            self.mean_reward = global_mean_reward
        else:
            self.mean_reward = (
                self.args.mean_baseline_coef * self.mean_reward
                + (1 - self.args.mean_baseline_coef) * global_mean_reward
            )

    def baseline_rewards(self, rewards):
        baseline = self.mean_reward if self.mean_reward is not None else 0
        advantages = rewards - baseline

        with torch.no_grad():
            metrics = {"baseline_mean": baseline}

        self.update_mean_reward(rewards=rewards)

        return advantages, metrics

    def get_log_mean_std(self, tensor, name, train_eval=None):
        mean = get_global_mean(tensor)
        metrics = {}
        if train_eval is not None:
            metrics[f"{train_eval}/{name}_mean"] = mean
            metrics[f"{train_eval}/{name}_std"] = get_global_std(tensor, mean=mean)
        else:
            metrics[f"{name}_mean"] = mean
            metrics[f"{name}_std"] = get_global_std(tensor, mean=mean)

        return metrics

    def _get_batch_loss_metrics(
        self, model, inputs, train_eval: Literal["train", "eval"] = "train"
    ):
        assert inputs["input_ids"].shape[1] <= self.max_tokens_count, (
            f"Input length {inputs['input_ids'].shape[1]} exceeds max_tokens_count {self.max_tokens_count}",
            self.tokenizer.decode(inputs["input_ids"][0]),
        )

        with torch.no_grad():
            (
                (
                    query_response,
                    attention_mask,
                    response_tokens_mask,
                    position_ids,
                    rewards,
                ),
                gen_metrics,
            ) = self.generate_responses(
                peft_policy=self.accelerator.unwrap_model(model),
                queries=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                train_eval=train_eval,
            )

            with model.disable_adapter():
                ref_logprobs = self.get_logprobs(
                    model=model,
                    input_ids=query_response,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    loss_mask=response_tokens_mask.to(torch.float32),
                )

            rewards, valid_mask, rewards_metrics = self.process_rewards(
                rewards=rewards, query_response=query_response
            )

        logprobs = self.get_logprobs(
            model=model,
            input_ids=query_response,
            attention_mask=attention_mask,
            position_ids=position_ids,
            loss_mask=response_tokens_mask,
        )

        with torch.no_grad():
            kl_term = logprobs.detach() - ref_logprobs
            regularized_rewards = rewards - self.kl_controller.value * kl_term

            baselined_reward, baseline_metrics = self.baseline_rewards(
                rewards=regularized_rewards
            )
            self.kl_controller.update(kl_term)

        loss = -baselined_reward * logprobs

        assert len(loss.shape) == 1, loss.shape

        with torch.no_grad():
            metrics = {}
            for tensor, name in zip(
                [
                    rewards,
                    regularized_rewards,
                    baselined_reward,
                    loss.detach(),
                    kl_term.detach(),
                    logprobs.detach(),
                    1 - valid_mask.float(),
                ],
                [
                    "rewards",
                    "regularized_rewards",
                    "baselined_rewards",
                    "loss",
                    "kl_term",
                    "logprobs",
                    "invalid_rewards",
                ],
            ):
                metrics = {
                    **metrics,
                    **self.get_log_mean_std(tensor, name, train_eval),
                }
            metrics[f"{train_eval}/kl_coef"] = self.kl_controller.value

            for k, v in rewards_metrics.items():
                metrics[f"{train_eval}/{k}"] = v
            for k, v in gen_metrics.items():
                metrics[f"{train_eval}/{k}"] = v
            for k, v in baseline_metrics.items():
                metrics[f"{train_eval}/{k}"] = v

        return loss.mean(), metrics

    def compute_loss(self, model, inputs, return_outputs: bool = False):
        loss, metrics = self._get_batch_loss_metrics(model, inputs, "train")

        self.store_metrics(metrics=metrics, train_eval="train")

        return (loss, metrics) if return_outputs else loss

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        with torch.no_grad():
            loss, metrics = self._get_batch_loss_metrics(model, inputs, "eval")

        self.store_metrics(metrics, train_eval="eval")

        return loss.detach(), None, None


class REINFORCETrainerVanillaRM(REINFORCETrainer):
    def postprocess_rewards(self, rewards):
        return rewards, {**self.get_log_mean_std(rewards, "real_reward")}


class REINFORCETrainerCategoricalPRM(REINFORCETrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.support = torch.linspace(-0.5, 0.5, 51).unsqueeze(0)

    def postprocess_rewards(self, rewards):
        probs = torch.nn.functional.softmax(rewards.to(torch.float32), dim=-1)
        mean = torch.sum(probs * self.support.to(rewards.device), -1)

        return mean.unsqueeze(-1), {
            **self.get_log_mean_std(mean, "real_reward"),
        }


class REINFORCETrainerCategoricalPRMVariancePenalty(REINFORCETrainer):
    def __init__(self, args, *aargs, **kwargs):
        super().__init__(args=args, *aargs, **kwargs)
        self.risk_coef = args.risk_coef

        self.support = torch.linspace(-0.5, 0.5, 51).unsqueeze(0)

    def postprocess_rewards(self, rewards):
        probs = torch.nn.functional.softmax(rewards.to(torch.float32), dim=-1)
        mean = torch.sum(probs * self.support.to(rewards.device), -1)
        variance = torch.sum(
            probs * (self.support.to(rewards.device) - mean.unsqueeze(-1)) ** 2, -1
        )

        return (mean - self.risk_coef * variance).unsqueeze(-1), {
            **self.get_log_mean_std(mean, "real_reward"),
            **self.get_log_mean_std(variance, "variance"),
        }


class REINFORCETrainerCategoricalPRMEntropyPenalty(REINFORCETrainer):
    def __init__(self, args, *aargs, **kwargs):
        super().__init__(args=args, *aargs, **kwargs)
        self.risk_coef = args.risk_coef

        self.support = torch.linspace(-0.5, 0.5, 51).unsqueeze(0)

    def postprocess_rewards(self, rewards):
        probs = torch.nn.functional.softmax(rewards.to(torch.float32), dim=-1)
        mean = torch.sum(probs * self.support.to(rewards.device), -1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), -1)

        return (mean - self.risk_coef * entropy).unsqueeze(-1), {
            **self.get_log_mean_std(mean, "real_reward"),
            **self.get_log_mean_std(entropy, "entropy"),
        }


class REINFORCETrainerRMRV(REINFORCETrainer):
    def postprocess_rewards(self, rewards):
        rewards, values = rewards[:, 0].unsqueeze(-1), rewards[:, 1].unsqueeze(-1)
        with torch.no_grad():
            metrics = {
                **self.get_log_mean_std(rewards, "real_reward"),
                **self.get_log_mean_std(values, "real_value"),
                **self.get_log_mean_std((rewards < values).float(), "frac_r<v"),
            }

        return rewards - values, metrics


class REINFORCETrainerRMRVNoEMA(REINFORCETrainerRMRV):
    def baseline_rewards(self, rewards):
        with torch.no_grad():
            metrics = {"baseline_mean": 0}

        return rewards, metrics


class REINFORCETrainerRMRVNoValues(REINFORCETrainer):
    def postprocess_rewards(self, rewards):
        rewards, values = rewards[:, 0].unsqueeze(-1), rewards[:, 1].unsqueeze(-1)
        with torch.no_grad():
            metrics = {
                **self.get_log_mean_std(rewards, "real_reward"),
                **self.get_log_mean_std(values, "real_value"),
                **self.get_log_mean_std((rewards < values).float(), "frac_r<v"),
            }

        return rewards, metrics


class RLOOTrainer(REINFORCETrainer):
    def postprocess_rewards(self, rewards):
        rewards = rewards[:, 0].unsqueeze(-1)  # values are at 1

        with torch.no_grad():
            metrics = self.get_log_mean_std(rewards, "real_reward")

        return rewards, metrics

    def baseline_rewards(self, rewards):
        rewards = rewards.reshape(-1, self.sampling_params.n)
        baseline = (rewards.sum(-1).unsqueeze(-1) - rewards) / (
            self.sampling_params.n - 1
        )
        rloo_advantages = (rewards - baseline).flatten()

        with torch.no_grad():
            metrics = {
                **self.get_log_mean_std(baseline, "baseline"),
                **self.get_log_mean_std(baseline.std(-1), "baseline_inner_std"),
            }

        return rloo_advantages, metrics
