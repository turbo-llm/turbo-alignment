from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import torch
import torch.nn.functional as F
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
from turbo_alignment.settings.pipelines.train.dpo import SyncRefModelSettings
from turbo_alignment.trainers.dpo import DPOTrainer
from turbo_alignment.trainers.utils import prepare_model

logger = get_project_logger()


@dataclass
class KTOTrainingArguments(TrainingArguments):
    beta: float = 0.1
    sync_ref_settings: SyncRefModelSettings = field(
        default_factory=SyncRefModelSettings()
    )  # type: ignore[call-overload]
    use_ref_model: bool = True
    average_log_prob: bool = False
    undesirable_weight: float = 1.0
    desirable_weight: float = 1.33


class KTOTrainer(DPOTrainer):
    """
    Inspired by https://github.com/huggingface/trl/blob/main/trl/trainer/kto_trainer.py

    """

    def __init__(
        self,
        model: PreTrainedModel | nn.Module,
        data_collator: Callable,
        args: KTOTrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        ref_model: PreTrainedModel | nn.Module | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        callbacks: list[TrainerCallback] | None = None,
        **kwargs,
    ):
        self.data_collator = data_collator

        self.average_log_prob = args.average_log_prob
        self.desirable_weight = args.desirable_weight
        self.undesirable_weight = args.undesirable_weight
        self.beta = args.beta

        self._stored_metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

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

        self.ref_model = ref_model

        if self.ref_model is not None:
            self.ref_model = prepare_model(self.ref_model, self.accelerator, self.is_deepspeed_enabled)

        default_callbacks = [DefaultFlowCallback] + get_reporting_integration_callbacks(self.args.report_to)
        callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = MetricsCallbackHandler(
            callbacks,
            model,
            tokenizer,
            None,
            None,
            ref_model=self.ref_model,
            accelerator=self.accelerator,
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else ProgressCallback)
        self.control: TrainerControl = self.callback_handler.on_init_end(self.args, self.state, self.control)

    def kto_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        policy_KL_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
        reference_KL_logps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        KL = (policy_KL_logps - reference_KL_logps).mean().detach()
        KL = self.accelerator.gather(KL).mean().clamp(min=0)

        if policy_chosen_logps.shape[0] != 0 or reference_chosen_logps.shape[0] != 0:
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            chosen_losses = 1 - F.sigmoid(self.beta * (chosen_logratios - KL))
            chosen_rewards = self.beta * chosen_logratios.detach()
        else:
            chosen_losses = torch.Tensor([]).to(self.accelerator.device)
            chosen_rewards = torch.Tensor([]).to(self.accelerator.device)

        if policy_rejected_logps.shape[0] != 0 or reference_rejected_logps.shape[0] != 0:
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            rejected_losses = 1 - F.sigmoid(self.beta * (KL - rejected_logratios))
            rejected_rewards = self.beta * rejected_logratios.detach()
        else:
            rejected_losses = torch.Tensor([]).to(self.accelerator.device)
            rejected_rewards = torch.Tensor([]).to(self.accelerator.device)

        losses = torch.cat(
            (self.desirable_weight * chosen_losses, self.undesirable_weight * rejected_losses),
            0,
        )

        return losses, chosen_rewards, rejected_rewards, KL

    def _forward(
        self, model: nn.Module, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            KL_logits = model(
                batch['KL_input_ids'],
                attention_mask=batch['KL_attention_mask'],
            ).logits

        completion_logits = model(
            batch['input_ids'],
            attention_mask=batch['attention_mask'],
        ).logits

        completion_logps = self._get_batch_logps(
            logits=completion_logits,
            labels=batch['labels'],
            average_log_prob=False,
        )

        KL_logps = self._get_batch_logps(
            logits=KL_logits,
            labels=batch['KL_labels'],
            average_log_prob=False,
        )

        return completion_logps, KL_logps, completion_logits, KL_logits

    def concatenated_forward(
        self, model: nn.Module, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        completion_logps, KL_logps, completion_logits, _ = self._forward(model, batch)

        if completion_logps.shape[0] != len(batch['is_desirable']):
            raise ValueError(
                'There is a mismatch between the number of examples in this batch and the number of '
                'examples for which an output sequence was predicted.'
            )

        chosen_idx = [i for i in range(completion_logps.shape[0]) if batch['is_desirable'][i]]
        rejected_idx = [i for i in range(completion_logps.shape[0]) if not batch['is_desirable'][i]]

        chosen_logps = completion_logps[chosen_idx, ...]
        rejected_logps = completion_logps[rejected_idx, ...]

        chosen_logits = completion_logits[chosen_idx, ...]
        rejected_logits = completion_logits[rejected_idx, ...]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, KL_logps)

    def _get_ref_logps(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            if self.ref_model is not None:
                reference_completion_logps, reference_KL_logps, *_ = self._forward(self.ref_model, batch)
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    (reference_completion_logps, reference_KL_logps, *_) = self._forward(self.model, batch)

        return reference_completion_logps, reference_KL_logps

    def get_batch_metrics(
        self,
        model: nn.Module,
        batch: dict[str, Any],
        train_eval: Literal['train', 'eval'] = 'train',
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        metrics = {}
        batch = {k: (v.to(self.accelerator.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            _,
            _,
            policy_KL_logps,
        ) = self.concatenated_forward(model, batch)

        with torch.no_grad():
            if self.ref_model is not None:
                reference_chosen_logps, reference_rejected_logps, _, _, reference_KL_logps = self.concatenated_forward(
                    self.ref_model, batch
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                        reference_KL_logps,
                    ) = self.concatenated_forward(self.model, batch)

        losses, chosen_rewards, rejected_rewards, kl = self.kto_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            policy_KL_logps=policy_KL_logps,
            reference_chosen_logps=reference_chosen_logps,
            reference_rejected_logps=reference_rejected_logps,
            reference_KL_logps=reference_KL_logps,
        )

        num_chosen = torch.Tensor([len(chosen_rewards)]).to(self.accelerator.device)
        num_rejected = torch.Tensor([len(rejected_rewards)]).to(self.accelerator.device)

        all_num_chosen = self.accelerator.gather(num_chosen)
        all_num_rejected = self.accelerator.gather(num_rejected)

        prefix = 'eval_' if train_eval == 'eval' else ''

        if all_num_chosen.sum().item() > 0:
            metrics[f'{prefix}rewards/chosen'] = (
                (self.accelerator.gather(chosen_rewards.mean()) * all_num_chosen).nansum() / all_num_chosen.sum()
            ).item()
            metrics[f'{prefix}logps/chosen'] = (
                (self.accelerator.gather(policy_chosen_logps.mean()) * all_num_chosen).nansum() / all_num_chosen.sum()
            ).item()

        if all_num_rejected.sum().item() > 0:
            metrics[f'{prefix}rewards/rejected'] = (
                (self.accelerator.gather(rejected_rewards.mean()) * all_num_rejected).nansum() / all_num_rejected.sum()
            ).item()
            metrics[f'{prefix}logps/rejected'] = (
                (self.accelerator.gather(policy_rejected_logps.mean()) * all_num_rejected).nansum()
                / all_num_rejected.sum()
            ).item()

        metrics[f'{prefix}kl'] = kl.item()
        if all_num_chosen.sum().item() > 0 and all_num_rejected.sum().item() > 0:
            metrics[f'{prefix}rewards/margins'] = (
                metrics[f'{prefix}rewards/chosen'] - metrics[f'{prefix}rewards/rejected']
            )

        del (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_KL_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            reference_KL_logps,
            kl,
        )
        del (
            num_chosen,
            num_rejected,
            all_num_chosen,
            all_num_rejected,
            chosen_rewards,
            rejected_rewards,
            _,
        )

        return losses.nanmean(), metrics

    def store_metrics(self, metrics: dict[str, float], train_eval: Literal['train', 'eval'] = 'train') -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)
            if isinstance(value, list):
                self._stored_metrics[train_eval][key].extend(value)
            else:
                self._stored_metrics[train_eval][key].append(value)
