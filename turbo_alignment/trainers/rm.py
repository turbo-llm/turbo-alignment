from typing import Any

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.trainer_pt_utils import nested_detach
from transformers.utils import logging

from turbo_alignment.trainers.multigpu import MultiGPUCherryPicksTrainer

logger = logging.get_logger(__name__)


class RMTrainer(MultiGPUCherryPicksTrainer):
    def compute_loss(self, model, inputs, return_outputs=False) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor:
        inputs_w = inputs['inputs_w']
        inputs_l = inputs['inputs_l']

        rewards_w = model(**inputs_w, return_dict=True)[0]
        rewards_l = model(**inputs_l, return_dict=True)[0]

        loss = -torch.nn.functional.logsigmoid(rewards_w - rewards_l).mean()
        if return_outputs:
            return loss, {'rewards_w': rewards_w, 'rewards_l': rewards_l}
        return loss

    def prediction_step(
        self,
        model: PreTrainedModel | nn.Module,
        inputs: dict[str, dict[str, torch.Tensor]],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, 'config'):
                ignore_keys = getattr(self.model.config, 'keys_to_ignore_at_inference', [])
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, logits_dict = self.compute_loss(model, inputs, return_outputs=True)

        if prediction_loss_only:
            return (loss, None, None)

        loss = loss.detach()
        logits = tuple(v for k, v in logits_dict.items() if k not in ignore_keys)
        logits = nested_detach(logits)
        logits = torch.stack(logits).mean(dim=2).T

        labels = logits[:, 0] > logits[:, 1]

        labels = labels.long()

        return loss, logits, labels
