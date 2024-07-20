from typing import Callable

import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
    TrainingArguments,
)

from turbo_alignment.trainers.multigpu import MultiGPUCherryPicksTrainer


class CustomLossTrainer(MultiGPUCherryPicksTrainer):
    def __init__(
        self,
        model: PreTrainedModel | nn.Module,
        args: TrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        custom_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        data_collator: DataCollator,
        tokenizer: PreTrainedTokenizerBase | None = None,
        callbacks: list[TrainerCallback] | None = None,
        model_init: Callable[[], PreTrainedModel] | None = None,
        **kwargs,
    ):
        self.custom_loss = custom_loss
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            callbacks=callbacks,
            **kwargs,
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Modified original version, without manual label smoothing
        """
        if 'labels' in inputs:
            labels = inputs.pop('labels')
        else:
            raise ValueError('No labels provided in the inputs')

        outputs = model(**inputs)
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs[0]

        loss = self.custom_loss(logits, labels)

        return (loss, outputs) if return_outputs else loss
