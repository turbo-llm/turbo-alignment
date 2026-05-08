import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from peft import PeftModel
from torch import nn
from transformers.modeling_utils import PreTrainedModel
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer_pt_utils import nested_detach
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import logging

from turbo_alignment.trainers.multigpu import MultiGPUCherryPicksTrainer
from turbo_alignment.modeling import parallel_states

logger = logging.get_logger(__name__)


class RMTrainer(MultiGPUCherryPicksTrainer):
    """
    Reward Model Trainer using sequential packing: [context | chosen | rejected].

    Processes all segments in a single forward pass. Extracts rewards at chosen_indices
    and rejected_indices, then computes ranking loss: -log(sigmoid(reward_chosen - reward_rejected)).

    Expected batch from PairPreferenceDataCollator:
        - 'input_ids': Sequentially packed sequences
        - 'attention_mask': 4D masks with chosen/rejected isolation
        - 'position_ids': Symmetric positions
        - 'chosen_indices', 'rejected_indices': Reward extraction positions
    """

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,  # pylint: disable=unused-argument
    ) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor:
        """
        Compute ranking loss from sequentially packed inputs.

        Extracts rewards at chosen_indices and rejected_indices from model outputs.

        Returns:
            loss or (loss, {'rewards_w': chosen_rewards, 'rewards_l': rejected_rewards})
        """

        device = self.accelerator.device
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        position_ids = inputs['position_ids'].to(device)

        attention_mask = torch.finfo(model.dtype).min * (attention_mask == 0).to(model.dtype)

        root_model = model.module if hasattr(model, 'module') else model

        # For PEFT models this resolves to the wrapped HF model
        wrapped_model = getattr(getattr(root_model, 'base_model', None), 'model', None) or root_model

        # Sequence-classification model keeps transformer backbone in `.model`
        backbone_model = getattr(wrapped_model, 'model', None)
        score_head = getattr(wrapped_model, 'score', None) or getattr(root_model, 'score', None)

        if backbone_model is None:
            raise AttributeError('Unable to find transformer backbone model for RM training')
        if score_head is None:
            raise AttributeError('Unable to find score head for RM training')

        outputs = backbone_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )

        if getattr(outputs, 'last_hidden_state', None) is not None:
            hidden_states = outputs.last_hidden_state
        elif isinstance(outputs, tuple) and len(outputs) > 0:
            hidden_states = outputs[0]
        else:
            raise AttributeError('Backbone output has no last_hidden_state')

        logits = score_head(hidden_states)  # [batch_size, seq_len, 1]

        chosen_indices = inputs['chosen_indices'].to(device)
        rejected_indices = inputs['rejected_indices'].to(device)

        if parallel_states.sequence_parallel_is_initialized():
            rank = parallel_states.get_sequence_parallel_rank()
            seq_len_chunk = logits.size(1)
            offset = rank * seq_len_chunk
            # logger.warning(

            def get_rewards(indices):
                is_local = (indices >= offset) & (indices < offset + seq_len_chunk)
                local_indices = indices - offset
                safe_indices = torch.where(is_local, local_indices, torch.zeros_like(local_indices))

                batch_idx = torch.arange(indices.size(0), device=logits.device)
                rewards = logits[batch_idx, safe_indices].squeeze(-1)
                rewards = rewards * is_local.to(rewards.dtype)

                dist.all_reduce(rewards, op=dist.ReduceOp.SUM, group=parallel_states.get_sequence_parallel_group())
                return rewards

            rewards_w = get_rewards(chosen_indices)
            rewards_l = get_rewards(rejected_indices)
        else:
            batch_size = input_ids.shape[0]
            rewards_w = logits[torch.arange(batch_size, device=logits.device), chosen_indices]
            rewards_l = logits[torch.arange(batch_size, device=logits.device), rejected_indices]

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
        logits = torch.stack(logits)
        if logits.dim() == 3:
            logits = logits.mean(dim=2)
        logits = logits.T

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
