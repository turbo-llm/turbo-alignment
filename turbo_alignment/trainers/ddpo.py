from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Literal

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    DefaultFlowCallback,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainingArguments,
)
from transformers.integrations import get_reporting_integration_callbacks

from turbo_alignment.common.logging import get_project_logger
from turbo_alignment.common.tf.callbacks.common import MetricsCallbackHandler
from turbo_alignment.trainers.dpo import DPOTrainer
from turbo_alignment.trainers.utils import concatenated_inputs, prepare_model

logger = get_project_logger()


@dataclass
class DDPOTrainingArguments(TrainingArguments):
    use_ref_model: bool = True
    average_log_prob: bool = False

    beta: float = 0.1
    forward_kl: bool = False


class DDPOTrainer(DPOTrainer):
    """
    Inspired by NLP Research

    """

    def __init__(
        self,
        model: PreTrainedModel | nn.Module,
        data_collator: Callable,
        args: DDPOTrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        rm: PreTrainedModel | nn.Module,
        ref_model: PreTrainedModel | nn.Module | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        callbacks: list[TrainerCallback] | None = None,
        **kwargs,
    ):
        self.data_collator = data_collator

        self.average_log_prob = args.average_log_prob

        self._stored_metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        self.beta = args.beta
        self.forward_kl = args.forward_kl

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            callbacks=callbacks,
            **kwargs,
        )

        default_callbacks = [DefaultFlowCallback] + get_reporting_integration_callbacks(self.args.report_to)
        callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = MetricsCallbackHandler(
            callbacks, model, tokenizer, None, None, ref_model=ref_model, accelerator=self.accelerator
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else ProgressCallback)
        self.control: TrainerControl = self.callback_handler.on_init_end(self.args, self.state, self.control)

        self.ref_model = ref_model
        self.rm = rm

        if self.ref_model is not None:
            self.ref_model = prepare_model(self.ref_model, self.accelerator, self.is_deepspeed_enabled)

        self.rm = prepare_model(self.rm, self.accelerator, self.is_deepspeed_enabled)

    def ddpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
        rm_reward_chosen: torch.Tensor,
        rm_reward_rejected: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        logits = pi_logratios - ref_logratios

        chosen_rewards = (self.beta * (policy_chosen_logps - reference_chosen_logps)).detach()
        rejected_rewards = (self.beta * (policy_rejected_logps - reference_rejected_logps)).detach()
        p_sft = torch.sigmoid(self.beta * logits)

        p_rm = torch.exp(rm_reward_chosen) / (torch.exp(rm_reward_chosen) + torch.exp(rm_reward_rejected))

        p1, p2 = p_sft, p_rm
        if self.forward_kl:
            p1, p2 = p_rm, p_sft

        loss = (1 - p1) * (torch.log(1 - p1) - torch.log(1 - p2)) + p1 * (torch.log(p1) - torch.log(p2))
        return loss, chosen_rewards, rejected_rewards

    def concatenated_sft_forward(
        self, model: nn.Module, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        concatenated_batch = concatenated_inputs(batch, prefix='sft_', device=self.accelerator.device)
        all_logits = model(
            concatenated_batch['input_ids'],
            attention_mask=concatenated_batch['attention_mask'],
        ).logits.to(torch.float32)
        all_logps = self._get_batch_logps(
            all_logits,
            concatenated_batch['labels'],
            average_log_prob=self.average_log_prob,
        )
        chosen_idxs = batch['sft_inputs_w']['input_ids'].shape[0]

        chosen_logps = all_logps[:chosen_idxs]
        rejected_logps = all_logps[chosen_idxs:]

        chosen_logits = all_logits[:chosen_idxs]
        rejected_logits = all_logits[chosen_idxs:]
        return chosen_logps, rejected_logps, chosen_logits, rejected_logits

    def concatenated_rm_forward(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        concatenated_batch = concatenated_inputs(batch, prefix='rm_', device=self.accelerator.device)
        with torch.no_grad():
            rewards = self.rm(
                input_ids=concatenated_batch['input_ids'], attention_mask=concatenated_batch['attention_mask']
            ).logits

        chosen_idxs = batch['rm_inputs_w']['input_ids'].shape[0]

        chosen_rewards, rejected_rewards = rewards[:chosen_idxs, :], rewards[chosen_idxs:, :]
        return chosen_rewards, rejected_rewards

    def _get_ref_logps(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            if self.ref_model is not None:
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    _,
                    _,
                ) = self.concatenated_sft_forward(self.ref_model, batch)
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                    ) = self.concatenated_sft_forward(self.model, batch)
        return reference_chosen_logps, reference_rejected_logps

    def get_batch_metrics(
        self,
        model: nn.Module,
        batch: dict[str, Any],
        train_eval: Literal['train', 'eval'] = 'train',
    ) -> tuple[torch.Tensor, dict[str, float]]:
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.concatenated_sft_forward(model, batch)

        reference_chosen_logps, reference_rejected_logps, *_ = self._get_ref_logps(batch)
        rm_rewards_chosen, rm_rewards_rejected = self.concatenated_rm_forward(batch)

        losses, chosen_rewards, rejected_rewards = self.ddpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            rm_rewards_chosen,
            rm_rewards_rejected,
        )

        sft_reward_accuracies = (chosen_rewards > rejected_rewards).float()
        rm_reward_accuracies = (rm_rewards_chosen > rm_rewards_rejected).float()

        prefix = 'eval_' if train_eval == 'eval' else ''
        metrics[f'{prefix}sft_rewards/chosen'] = chosen_rewards.detach().cpu().mean().item()
        metrics[f'{prefix}sft_rewards/rejected'] = rejected_rewards.detach().cpu().mean().item()
        metrics[f'{prefix}sft_rewards/accuracies'] = sft_reward_accuracies.detach().cpu().mean().item()
        metrics[f'{prefix}sft_rewards/margins'] = (chosen_rewards - rejected_rewards).detach().cpu().mean().item()

        metrics[f'{prefix}rm_rewards/chosen'] = rm_rewards_chosen.detach().cpu().mean().item()
        metrics[f'{prefix}rm_rewards/rejected'] = rm_rewards_rejected.detach().cpu().mean().item()
        metrics[f'{prefix}rm_rewards/accuracies'] = rm_reward_accuracies.detach().cpu().mean().item()
        metrics[f'{prefix}rm_rewards/margins'] = (rm_rewards_chosen - rm_rewards_rejected).detach().cpu().mean().item()

        metrics[f'{prefix}logps/rejected'] = policy_rejected_logps.detach().cpu().mean().item()
        metrics[f'{prefix}logps/chosen'] = policy_chosen_logps.detach().cpu().mean().item()
        metrics[f'{prefix}logits/rejected'] = policy_rejected_logits.detach().cpu().mean().item()
        metrics[f'{prefix}logits/chosen'] = policy_chosen_logits.detach().cpu().mean().item()

        return losses.nanmean(), metrics
