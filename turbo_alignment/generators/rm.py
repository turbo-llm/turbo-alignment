from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase

from turbo_alignment.dataset.pair_preferences import PairPreferenceDataCollator, PairPreferenceRecord
from turbo_alignment.dataset.sampling.models import SamplingDatasetRecord
from turbo_alignment.generators.base import BaseGenerator
from turbo_alignment.modeling import parallel_states
from turbo_alignment.sequence_parallel.collator import DataCollatorForSequenceParallism
from turbo_alignment.settings.generators.outputs.rm import (
    RMPairInferenceOutput,
    RMSamplingInferenceOutput,
)


class RMPairGenerator(BaseGenerator[PairPreferenceRecord, RMPairInferenceOutput]):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, **kwargs):
        collator: Any = PairPreferenceDataCollator(tokenizer=tokenizer, add_labels=False)
        if parallel_states.sequence_parallel_is_initialized():
            collator = DataCollatorForSequenceParallism.create_with_tokenizer(
                collator,
                seq_p_rank=parallel_states.get_sequence_parallel_rank(),
                seq_p_world_size=parallel_states.get_sequence_parallel_world_size(),
                tokenizer=tokenizer,
                fields_not_to_split=['attention_mask', 'chosen_indices', 'rejected_indices'],
            )

        self._collator = collator
        super().__init__(tokenizer=tokenizer, **kwargs)

    def _generate_from_batch(
        self, records: list[dict[str, Any]], original_records: list[PairPreferenceRecord], dataset_name: str
    ) -> list[RMPairInferenceOutput]:
        batch = self._collator(records)

        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        position_ids = batch['position_ids'].to(self.device)
        chosen_indices = batch['chosen_indices'].to(self.device)
        rejected_indices = batch['rejected_indices'].to(self.device)

        root_model = self._model.module if hasattr(self._model, 'module') else self._model
        attention_mask = torch.finfo(root_model.dtype).min * (attention_mask == 0).to(root_model.dtype)

        wrapped_model = getattr(getattr(root_model, 'base_model', None), 'model', None) or root_model
        backbone_model = getattr(wrapped_model, 'model', None)
        score_head = getattr(wrapped_model, 'score', None) or getattr(root_model, 'score', None)

        if backbone_model is None:
            raise AttributeError('Unable to find transformer backbone model for RM generation')
        if score_head is None:
            raise AttributeError('Unable to find score head for RM generation')

        with torch.no_grad():
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

            logits = score_head(hidden_states)

            if parallel_states.sequence_parallel_is_initialized():
                rank = parallel_states.get_sequence_parallel_rank()
                seq_len_chunk = logits.size(1)
                offset = rank * seq_len_chunk

                def get_rewards(indices: torch.Tensor) -> torch.Tensor:
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
                batch_idx = torch.arange(input_ids.shape[0], device=logits.device)
                rewards_w = logits[batch_idx, chosen_indices].squeeze(-1)
                rewards_l = logits[batch_idx, rejected_indices].squeeze(-1)

        return [
            RMPairInferenceOutput(
                id=record.id,
                context=record.context,
                answer_w=record.answer_w,
                answer_l=record.answer_l,
                reward_w=reward_w.item(),
                reward_l=reward_l.item(),
                dataset_name=dataset_name,
            )
            for record, reward_w, reward_l in zip(original_records, rewards_w.cpu(), rewards_l.cpu())
        ]


class RMSamplingGenerator(BaseGenerator[SamplingDatasetRecord, RMSamplingInferenceOutput]):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, micro_batch: int, **kwargs):
        self._collator = DataCollatorWithPadding(tokenizer=tokenizer)
        self._micro_batch = micro_batch
        super().__init__(tokenizer=tokenizer, **kwargs)

    def _generate_from_batch(
        self, records: list[dict[str, Any]], original_records: list[SamplingDatasetRecord], dataset_name: str
    ) -> list[RMSamplingInferenceOutput]:
        merged_inputs = [inputs for record in records for key, inputs in record['answers'].items()]

        if len(merged_inputs) == 0:
            return []

        rewards = []
        with torch.no_grad():
            input_ids = nn.utils.rnn.pad_sequence(
                [item['input_ids'] for item in merged_inputs],
                padding_value=self._tokenizer.pad_token_id,
                batch_first=True,
                padding_side=self._tokenizer.padding_side,
            )
            attn_mask = nn.utils.rnn.pad_sequence(
                [item['attention_mask'] for item in merged_inputs],
                padding_value=0,
                batch_first=True,
                padding_side=self._tokenizer.padding_side,
            )
            for i in range(0, len(input_ids), self._micro_batch):
                input_ids_batch = input_ids[i : i + self._micro_batch].to(self.device)
                attn_mask_batch = attn_mask[i : i + self._micro_batch].to(self.device)
                rewards.extend(self._model(input_ids=input_ids_batch, attention_mask=attn_mask_batch).logits.cpu())

        rewards = torch.cat(rewards, dim=0)

        reward_index = 0
        record_rewards = []
        for record in records:
            mapped_rewards = {}
            for key in record['answers'].keys():
                mapped_rewards[key] = rewards[reward_index].item()
                reward_index += 1
            record_rewards.append(mapped_rewards)

        return [
            RMSamplingInferenceOutput(
                id=record.id,
                rewards=rewards,
                messages=record.messages,
                dataset_name=dataset_name,
                answers=record.answers,
            )
            for record, rewards in zip(original_records, record_rewards)
        ]
