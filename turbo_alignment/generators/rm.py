from typing import Any

import torch
from torch import nn
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase

from turbo_alignment.dataset.pair_preferences import PairPreferenceRecord
from turbo_alignment.dataset.sampling.models import SamplingDatasetRecord
from turbo_alignment.generators.base import BaseGenerator
from turbo_alignment.settings.generators.outputs.rm import (
    RMPairInferenceOutput,
    RMSamplingInferenceOutput,
)


class RMPairGenerator(BaseGenerator[PairPreferenceRecord, RMPairInferenceOutput]):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, **kwargs):
        self._collator = DataCollatorWithPadding(tokenizer=tokenizer)

        super().__init__(tokenizer=tokenizer, **kwargs)

    def _generate_from_batch(
        self, records: list[dict[str, Any]], original_records: list[PairPreferenceRecord], dataset_name: str
    ) -> list[RMPairInferenceOutput]:
        merged_inputs = [r['inputs_w'] for r in records] + [r['inputs_l'] for r in records]
        batch = self._collator(merged_inputs)
        input_ids = batch['input_ids'].to(self.device)
        attn_mask = batch['attention_mask'].to(self.device)

        with torch.no_grad():
            rewards = self._model(input_ids=input_ids, attention_mask=attn_mask).logits.cpu()
            rewards_w, rewards_l = rewards[: len(records)], rewards[len(records) :]

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
            for record, reward_w, reward_l in zip(original_records, rewards_w, rewards_l)
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
                padding_side='left',
                # padding_side=self._tokenizer.padding_side,
            )
            attn_mask = nn.utils.rnn.pad_sequence(
                [item['attention_mask'] for item in merged_inputs],
                padding_value=0,
                batch_first=True,
                padding_side='left',
                # padding_side=self._tokenizer.padding_side,
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
