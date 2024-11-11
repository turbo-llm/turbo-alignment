from pathlib import Path
from typing import Any, Callable, Literal

import gc

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
)

from turbo_alignment.common.data.io import write_jsonl
from turbo_alignment.common.logging import get_project_logger
from turbo_alignment.constants import DISABLE_LOSS_LABEL
from turbo_alignment.trainers.utils import (
    concatenated_inputs,
    prepare_model,
)

logger = get_project_logger()



class ref_model_DPOTrainer(Trainer):
    """
    Inspired by https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py

    """

    def __init__(
        self,
        model: PreTrainedModel | nn.Module,
        data_collator: Callable,
        args,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        ref_model: PreTrainedModel | nn.Module | None = None,
        sft_model: PreTrainedModel | nn.Module | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        callbacks: list[TrainerCallback] | None = None,
        **kwargs,
    ):
        self.data_collator = data_collator

        self.output_path = kwargs.pop('output_path')

        self.average_log_prob = args.average_log_prob

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
        self.model = prepare_model(ref_model, self.accelerator, self.is_deepspeed_enabled)

      
    def _get_batch_logps(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        average_log_prob: bool = False,
    ) -> torch.Tensor:
        if logits.shape[:-1] != labels.shape:
            raise ValueError('Logits (batch and sequence length dim) and labels must have the same shape.')

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != DISABLE_LOSS_LABEL

        labels[labels == DISABLE_LOSS_LABEL] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        return (per_token_logps * loss_mask).sum(-1)

    def concatenated_forward(
        self, model: nn.Module, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        concatenated_batch = concatenated_inputs(batch, device=self.accelerator.device)

        precomputed_margins: torch.Tensor | None = concatenated_batch.pop('margin', None)

        all_logits = model(
            concatenated_batch['input_ids'],
            attention_mask=concatenated_batch['attention_mask'],
        ).logits.to(torch.float32)
        all_logps = self._get_batch_logps(
            all_logits,
            concatenated_batch['labels'],
            average_log_prob=self.average_log_prob,
        )
        chosen_idxs = batch['inputs_w']['input_ids'].shape[0]
        rejected_idx = batch['inputs_l']['input_ids'].shape[0]

        chosen_logps = all_logps[:chosen_idxs]
        rejected_logps = all_logps[chosen_idxs : chosen_idxs + rejected_idx]

        chosen_logits = all_logits[:chosen_idxs]
        rejected_logits = all_logits[chosen_idxs:]
        return chosen_logps, rejected_logps, chosen_logits, rejected_logits, precomputed_margins

    def _get_logps(self, model: nn.Module | None, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            if model is not None:
                (chosen_logps, rejected_logps, *_) = self.concatenated_forward(model, batch)
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    (
                        chosen_logps,
                        rejected_logps,
                        *_,
                    ) = self.concatenated_forward(self.model, batch)

        return chosen_logps, rejected_logps

    def get_batch_metrics(
        self,
        model: nn.Module,
        batch: dict[str, Any],
        train_eval: Literal['train', 'eval'] = 'train',
    ) -> tuple[torch.Tensor, dict[str, float]]:

        reference_chosen_logps, reference_rejected_logps = self._get_logps(self.model, batch)
        return reference_chosen_logps, reference_rejected_logps

    def compute_loss(
        self,
        model: PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs=False,
        num_items_in_batch=None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        reference_chosen_logps, reference_rejected_logps = self.get_batch_metrics(self.model, inputs, train_eval='train')

        if self.accelerator.is_main_process:
            data = {
                'inputs_w': inputs['inputs_w']['input_ids'].tolist(),
                'inputs_l': inputs['inputs_l']['input_ids'].tolist(),
                'chosen_logps': reference_chosen_logps.tolist(),
                'rejected_logps': reference_rejected_logps.tolist(),
            }
            write_jsonl([data], Path(self.output_path), 'a+')

        gc.collect()
        torch.cuda.empty_cache()
        return torch.tensor([0.0], requires_grad=True)

    def prediction_step(
        self,
        model: PreTrainedModel | nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:

        if prediction_loss_only:
            return torch.tensor([0.0]).detach(), None, None

        return torch.tensor([0.0]).detach(), torch.tensor([0.0]), torch.tensor([0.0])
