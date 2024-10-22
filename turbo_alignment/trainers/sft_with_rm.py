from typing import Any
from pathlib import Path
import os

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.trainer_pt_utils import nested_detach
from transformers.utils import logging

from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from turbo_alignment.trainers.multigpu import MultiGPUCherryPicksTrainer

logger = logging.get_logger(__name__)


class SFTwithRMTrainer(MultiGPUCherryPicksTrainer):
    def compute_loss(self, model, inputs, return_outputs=False) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor:
        sft_logits, rewards_w, rewards_l, reward_token_pos_w = model.forward(inputs)

        sft_logits = sft_logits.view(-1, sft_logits.size(-1))
        sft_labels = inputs['inputs_w']['input_ids']

        sft_labels_1 = sft_labels.view(-1)[: reward_token_pos_w[0]]
        sft_labels_2 = sft_labels.view(-1)[reward_token_pos_w[0] + 1 :]
        sft_labels_cat = torch.cat((sft_labels_1, sft_labels_2), dim=0)

        loss = -torch.nn.functional.logsigmoid(rewards_w - rewards_l).mean() + torch.nn.functional.cross_entropy(
            sft_logits, sft_labels_cat
        )
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

        logits = torch.stack(logits).T

        labels = logits[:, 0] > logits[:, 1]

        labels = labels.long()

        return loss, logits, labels

    def _save_checkpoint(self, model, trial, metrics=None):
        logger.info('Running custom _save_checkpoint')
        checkpoint_folder = f'{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}'
        run_dir = self._get_output_dir(trial=trial)
        output_dir = Path(os.path.join(run_dir, checkpoint_folder))

        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith('eval_'):
                metric_to_check = f'eval_{metric_to_check}'
            metric_value = metrics[metric_to_check]
            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        (output_dir / 'decoder').mkdir(parents=True, exist_ok=True)

        torch.save(model.module.lm_head.state_dict(), output_dir / 'lm_head.pt')
        torch.save(model.module.rm_head.state_dict(), output_dir / 'rm_head.pt')

        model.module.decoder.save_pretrained(output_dir / 'decoder')
        self.tokenizer.save_pretrained(output_dir / 'decoder')
