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
    alpha: float = 0.1


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
        self.alpha = args.alpha

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

        per_token_logps = self._get_batch_logps(all_logits, concatenated_batch['labels'])
        chosen_idxs = batch['inputs_w']['input_ids'].shape[0]
        rejected_idx = batch['inputs_l']['input_ids'].shape[0]

        chosen_logits = all_logits[:chosen_idxs]
        rejected_logits = all_logits[chosen_idxs:]

        chosen_per_token_logps = per_token_logps[:chosen_idxs]
        rejected_per_token_logps = per_token_logps[chosen_idxs : chosen_idxs + rejected_idx]

        chosen_loss_mask = loss_mask[:chosen_idxs]
        rejected_loss_mask = loss_mask[chosen_idxs : chosen_idxs + rejected_idx]

        min_lengths = torch.min(chosen_loss_mask.sum(-1), rejected_loss_mask.sum(-1))

        answer_start_idx = torch.argmax(
            chosen_loss_mask.int(), -1
        )  # The index of the beginning of the chosen and rejected
        split_start_idx = answer_start_idx + min_lengths  # Add the length of the shorter answer

        # Setting the increment mask
        alpha_mask = torch.arange(torch.full_like(chosen_loss_mask, 1).size(1)).unsqueeze(
            0
        ) >= split_start_idx.unsqueeze(1)

        # Incrementing by alpha logprobs that are out of bounds
        chosen_per_token_logps[alpha_mask] = chosen_per_token_logps[alpha_mask] ** self.alpha
        rejected_per_token_logps[alpha_mask] = rejected_per_token_logps[alpha_mask] ** self.alpha

        if self.average_log_prob:
            chosen_logps = (chosen_per_token_logps * chosen_loss_mask).sum(-1) / chosen_loss_mask.sum(-1)
            rejected_logps = (rejected_per_token_logps * rejected_loss_mask).sum(-1) / rejected_loss_mask.sum(-1)
        chosen_logps = (chosen_per_token_logps * chosen_loss_mask).sum(-1)
        rejected_logps = (rejected_per_token_logps * rejected_loss_mask).sum(-1)

        return chosen_logps, rejected_logps, chosen_logits, rejected_logits, precomputed_margins
