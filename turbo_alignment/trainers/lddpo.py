from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback

from turbo_alignment.constants import DISABLE_LOSS_LABEL
from turbo_alignment.trainers.dpo import DPOTrainer, DPOTrainingArguments
from turbo_alignment.trainers.utils import concatenated_inputs


@dataclass
class LDDPOTrainingArguments(DPOTrainingArguments):
    lc_alpha: float = 0.1


class LDDPOTrainer(DPOTrainer):
    """
    From https://arxiv.org/pdf/2409.06411

    """

    def __init__(
        self,
        model: PreTrainedModel | nn.Module,
        data_collator: Callable,
        args: LDDPOTrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        tokenizer: PreTrainedTokenizerBase | None = None,
        callbacks: list[TrainerCallback] | None = None,
        **kwargs,
    ):
        self.lc_alpha = args.lc_alpha

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

    def _get_batch_logps(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits.shape[:-1] != labels.shape:
            raise ValueError('Logits (batch and sequence length dim) and labels must have the same shape.')

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]

        labels[labels == DISABLE_LOSS_LABEL] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        return per_token_logps

    def concatenated_forward(
        self, model: nn.Module, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        concatenated_batch = concatenated_inputs(batch, device=self.accelerator.device)

        precomputed_margins: torch.Tensor | None = concatenated_batch.pop('margin', None)

        all_logits = model(
            concatenated_batch['input_ids'],
            attention_mask=concatenated_batch['attention_mask'],
        ).logits.to(torch.float32)

        loss_mask = concatenated_batch['labels'][:, 1:] != DISABLE_LOSS_LABEL
        batch_size = concatenated_batch['input_ids'].size(0) // 2
        chosen_mask, rejected_mask = loss_mask.split(batch_size, dim=0)

        all_logps = self._get_batch_logps(all_logits, concatenated_batch['labels'])

        public_ = chosen_mask * rejected_mask
        public_mask = torch.cat([public_, public_])
        public_logps = all_logps * public_mask
        all_logps = self.lc_alpha * all_logps + (1 - self.lc_alpha) * public_logps

        chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)
        chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)

        return chosen_logps, rejected_logps, chosen_logits, rejected_logits, precomputed_margins
