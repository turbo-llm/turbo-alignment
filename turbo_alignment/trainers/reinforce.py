from dataclasses import dataclass
from typing import Literal

import ray
import torch
import torch.distributed
import torch.utils.data
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback

from turbo_alignment.dist_utils.ray.rayactor_group import RayGroup
from turbo_alignment.dist_utils.utils import get_log_mean_std
from turbo_alignment.settings.pipelines.train.online import RewardProcessorType
from turbo_alignment.trainers.online import OnlineTrainer, OnlineTrainingArguments


# FIXME
@dataclass
class REINFORCETrainingArguments(OnlineTrainingArguments):
    reward_processor_type: RewardProcessorType = RewardProcessorType.RLOO


class REINFORCETrainer(OnlineTrainer):
    def __init__(
        self,
        vllm_engines: list,
        args: REINFORCETrainingArguments,
        processing_class: PreTrainedTokenizerBase,
        policy: PreTrainedModel | torch.nn.Module,
        reward_model: PreTrainedModel | torch.nn.Module,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        ref_model: PreTrainedModel | torch.nn.Module = None,
        callbacks: list[TrainerCallback] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            vllm_engines=vllm_engines,
            args=args,
            processing_class=processing_class,
            policy=policy,
            reward_model=reward_model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            ref_model=ref_model,
            callbacks=callbacks,
            **kwargs,
        )

    def get_batch_loss_metrics(
        self,
        model: torch.nn.Module | PreTrainedModel,
        inputs: dict[str, torch.Tensor],
        train_eval: Literal['train', 'eval'] = 'train',
    ):
        with torch.no_grad():
            (
                query_response,
                attention_mask,
                response_tokens_mask,
                position_ids,
                rewards,
            ) = self.get_answers_and_rewards(
                model=model,
                inputs=inputs,
                do_broadcast=(train_eval == 'train'),
            )

            if isinstance(self.ref_model, RayGroup):
                ref_logits = ray.get(
                    self.ref_model.reference_forward(
                        {
                            'input_ids': query_response,
                            'attention_mask': attention_mask,
                            'position_ids': position_ids,
                        },
                        index=torch.distributed.get_rank() % self.args.reference_model_replicas,
                    )
                )
            else:
                ref_logits = self.ref_model(  # type: ignore[misc]
                    input_ids=query_response,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                ).logits[:, :-1]

            ref_logprobs = self.get_logprobs(
                logits=ref_logits,
                input_ids=query_response,
                loss_mask=response_tokens_mask,
                do_sum=False,
            )

            rewards, valid_mask, rewards_metrics = self.process_rewards(rewards=rewards, query_response=query_response)

        logits = model(
            input_ids=query_response,
            attention_mask=attention_mask,
            position_ids=position_ids,
        ).logits[:, :-1]

        # del inputs
        # gc.collect()
        # torch.cuda.empty_cache() #NEED?

        logprobs = self.get_logprobs(
            logits=logits,
            input_ids=query_response,
            loss_mask=response_tokens_mask,
            do_sum=False,
        )

        with torch.no_grad():
            kl_term = logprobs.detach() - ref_logprobs
            regularized_rewards = rewards - self.kl_coef * kl_term

            advantages, advantages_metrics = self.reward_processor.baseline_rewards(
                rewards=regularized_rewards
            )  # its advantage

        loss = -advantages * logprobs

        assert len(loss.shape) == 1, loss.shape

        with torch.no_grad():
            metrics: dict[str, float] = {}
            for tensor, name in zip(
                [
                    rewards,
                    advantages,
                    loss.detach(),
                    kl_term.mean().detach(),
                    logprobs.detach(),
                    ref_logprobs,
                    1 - valid_mask.float(),
                ],
                [
                    'rewards',
                    'advantages',
                    'loss',
                    'kl_term',
                    'logprobs',
                    'ref_logprobs',
                    'invalid_rewards',
                ],
            ):
                metrics = {
                    **metrics,
                    **get_log_mean_std(tensor.cuda(), name, train_eval),
                }

            for k, v in rewards_metrics.items():
                metrics[f'{train_eval}/{k}'] = v
            for k, v in advantages_metrics.items():
                metrics[f'{train_eval}/{k}'] = v

            for k, v in metrics.items():
                if isinstance(v, torch.Tensor):
                    metrics[k] = metrics[k]

        return loss.mean(), metrics
