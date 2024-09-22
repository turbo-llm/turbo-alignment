from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils
import torch.utils.data
from datasets import Dataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainerCallback,
    TrainingArguments,
)

from turbo_alignment.trainers.multigpu import MultiGPUCherryPicksTrainer
from turbo_alignment.common.distributed import get_global_mean, get_global_std, get_log_mean_std
from turbo_alignment.trainers.utils import prepare_model
from turbo_alignment.trainers.online.reward_processor import RewardProcessor
from turbo_alignment.trainers.online.llm_actor import LLMActor


@dataclass
class REINFORCETrainingArguments(TrainingArguments):
    max_tokens_count: int
    stop_token: str

    penalty_reward_value: float
    clip_rewards_min: float
    clip_rewards_max: float
    kl_coef: float
    mean_baseline_coef: float

    num_generations: int = 1
    num_samples_for_reward_stats: int = 0

    non_eos_penalty: bool = True
    temperature: float | None = None
    whiten_rewards: bool = False


class REINFORCETrainer(MultiGPUCherryPicksTrainer):

    def __init__(
        self,
        args: REINFORCETrainingArguments,
        tokenizer: PreTrainedTokenizer,
        policy: PreTrainedModel | torch.nn.Module,
        ref_model: PreTrainedModel | torch.nn.Module,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        callbacks: list[TrainerCallback] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model=policy,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            callbacks=callbacks,
            **kwargs,
        )

        self.ref_model = ref_model
        if self.ref_model is not None:
            self.ref_model = prepare_model(self.ref_model, self.accelerator, self.is_deepspeed_enabled)

        self._stored_metrics: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self.kl_coef = args.kl_coef

        self.gen_id = 1

        # Take gradient accumulation into account
        # In this way, the effective number of gradient
        # updates taken into account stays the same
        effective_num_previous_samples = 1 / (1 - self.args.mean_baseline_coef)
        self.args.mean_baseline_coef = 1 - 1 / (
            effective_num_previous_samples * self.args.gradient_accumulation_steps
        )

        self.norm_reward_mean, self.norm_reward_std = self.reward_stats(
            model=self.model, dataloader=self.get_train_dataloader()
        )

        stop_generation_token_id = tokenizer.encode(
            self.args.stop_token, add_special_tokens=False
        )
        assert len(stop_generation_token_id) == 1, stop_generation_token_id

        # FIXME: registry
        self.llm_actor = LLMActor(
            max_tokens_count=args.max_tokens_count,
            stop_token_id=stop_generation_token_id,
            temperature=args.temperature,
        )
        self.reward_processor = RewardProcessor(mean_baseline_coef=args.mean_baseline_coef)

    def get_batch_loss_metrics(
        self, 
        model: torch.nn.Module | PreTrainedModel, 
        inputs: torch.Tensor, 
        train_eval: Literal["train", "eval"] = "train",
    ):
        assert inputs["input_ids"].shape[1] <= self.args.max_tokens_count, (
            f"Input length {inputs['input_ids'].shape[1]} exceeds max_tokens_count {self.args.max_tokens_count}",
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
            ) = self.llm_actor.generate_responses(
                model=self.accelerator.unwrap_model(model),
                queries=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                train_eval=train_eval,
            )

            ref_logprobs = self.get_logprobs(
                model=self.ref_model,
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
            regularized_rewards = rewards - self.kl_coef * kl_term

            baselined_reward, baseline_metrics = self.reward_processor.baseline_rewards(
                rewards=regularized_rewards
            )

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
                    **get_log_mean_std(tensor, name, train_eval),
                }
            metrics[f"{train_eval}/kl_coef"] = self.kl_coef

            for k, v in rewards_metrics.items():
                metrics[f"{train_eval}/{k}"] = v
            for k, v in gen_metrics.items():
                metrics[f"{train_eval}/{k}"] = v
            for k, v in baseline_metrics.items():
                metrics[f"{train_eval}/{k}"] = v

        return loss.mean(), metrics

    def compute_loss(self, model, inputs, return_outputs: bool = False):
        loss, metrics = self.get_batch_loss_metrics(model, inputs, "train")

        self.store_metrics(metrics=metrics, train_eval="train")

        return (loss, metrics) if return_outputs else loss

    def prediction_step(
        self,
        model: Union[PreTrainedModel, torch.nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        with torch.no_grad():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, "eval")

        self.store_metrics(metrics, train_eval="eval")

        return loss.detach(), None, None

    @torch.no_grad()
    def reward_stats(
        self, 
        model: torch.nn.Module | PreTrainedModel, 
        dataloader: torch.utils.data.Dataloader
    ) -> tuple[float, float]:
        if self.args.num_samples_for_reward_stats == 0:
            norm_reward_mean = 0
            norm_reward_std = 1
        else:
            all_rewards = []
            samples_processed = 0
            for batch in dataloader:
                (_, _, _, _, rewards), _ = self.llm_actor.generate_responses(
                    model=model,
                    queries=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    train_eval="eval",
                )
                rewards, _ = self.reward_processor.postprocess_rewards(rewards=rewards)

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

        logprob = torch.gather(all_logprob, 2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        logprob[~loss_mask[:, 1:].to(torch.bool)] = 0

        return logprob.sum(-1)

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

    def process_rewards(self, rewards, query_response) -> tuple[torch.Tensor, torch.Tensor, Any]:
        """
        Second, this token should be the <eot_id>.
        Otherwise, return a fixed reward of -1.
        """

        rewards, reward_metrics = self.reward_processor.postprocess_rewards(rewards=rewards)
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
