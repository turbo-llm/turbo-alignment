import os
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from torch import nn
from transformers.modeling_utils import PreTrainedModel
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer_pt_utils import nested_detach
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import logging

from turbo_alignment.trainers.multigpu import MultiGPUCherryPicksTrainer
from turbo_alignment.trainers.utils import concatenated_inputs
from turbo_alignment.sequence_parallel.collator import pad_for_sequence_parallel
from turbo_alignment.modeling import parallel_states

logger = logging.get_logger(__name__)


class RMTrainer(MultiGPUCherryPicksTrainer):
    def concatenated_forward(self, model: nn.Module, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        concatenated_batch = concatenated_inputs(batch, device=self.accelerator.device)

        input_ids = concatenated_batch['input_ids']
        attention_mask = concatenated_batch['attention_mask']

        if parallel_states.sequence_parallel_is_initialized():
            input_ids = pad_for_sequence_parallel(
                input_ids,
                parallel_states.get_sequence_parallel_world_size(),
                self.tokenizer.pad_token_id,
                padding_side=self.tokenizer.padding_side,
            )
            attention_mask = pad_for_sequence_parallel(
                attention_mask,
                parallel_states.get_sequence_parallel_world_size(),
                0,
                padding_side=self.tokenizer.padding_side,
            )

            chunk_size = input_ids.size(-1) // parallel_states.get_sequence_parallel_world_size()
            start = chunk_size * parallel_states.get_sequence_parallel_rank()
            end = chunk_size * (parallel_states.get_sequence_parallel_rank() + 1)
            input_ids = input_ids[:, start:end].clone()

        all_rewards = model(input_ids, attention_mask=attention_mask, return_dict=True)[0]

        chosen_idxs = batch['inputs_w']['input_ids'].shape[0]

        chosen_rewards = all_rewards[:chosen_idxs]
        rejected_rewards = all_rewards[chosen_idxs:]

        return chosen_rewards, rejected_rewards

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,  # pylint: disable=unused-argument
    ) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor:
        rewards_w, rewards_l = self.concatenated_forward(model, inputs)

        loss = -torch.nn.functional.logsigmoid(rewards_w - rewards_l).mean()
        if return_outputs:
            return loss, {'rewards_w': rewards_w, 'rewards_l': rewards_l}
        return loss

    def prediction_step(  # type: ignore[override]  #  pylint: disable=signature-differs
        self,
        model: PreTrainedModel | nn.Module,
        inputs: dict[str, dict[str, torch.Tensor]],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        inputs = self._prepare_inputs(inputs)  # type: ignore[arg-type]
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

    def _save_checkpoint(self, model, trial):
        if isinstance(model, PeftModel) and is_deepspeed_zero3_enabled():
            logger.info('Running custom _save_checkpoint')
            checkpoint_folder = f'{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}'
            run_dir = self._get_output_dir(trial=trial)
            output_dir = Path(os.path.join(run_dir, checkpoint_folder))

            (output_dir / 'cls_head').mkdir(parents=True, exist_ok=True)

            torch.save(model.base_model.model.score.state_dict(), output_dir / 'cls_head' / 'cls_head.pt')

        return super()._save_checkpoint(model=model, trial=trial)  # pylint: disable=no-member
