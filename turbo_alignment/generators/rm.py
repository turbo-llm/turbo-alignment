from typing import Any

import torch
from transformers import BatchEncoding, DataCollatorWithPadding, PreTrainedTokenizerBase

from turbo_alignment.dataset.sampling.models import SamplingDatasetRecord
from turbo_alignment.generators.base import BaseGenerator
from turbo_alignment.settings.generators.outputs.rm import RMSamplingInferenceOutput
import ray

class RayRMSamplingGenerator(BaseGenerator[SamplingDatasetRecord, RMSamplingInferenceOutput]):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, micro_batch: int = 1, model_replicas: int = 1, **kwargs):
        self._collator = DataCollatorWithPadding(tokenizer=tokenizer)
        self._micro_batch = micro_batch
        self.model_replicas = model_replicas
        super().__init__(tokenizer=tokenizer, **kwargs)

    def generate_from_batch_records(self, records: dict[str, torch.Tensor] | BatchEncoding) -> torch.Tensor:
        with torch.no_grad():
            # TODO assuming that only one reward model
            records = {k: v.cuda() for k, v in records.items()}
            
            rank = torch.distributed.get_rank()

            rewards = ray.get(self._model.reward_forward(records=records, index=rank % self.model_replicas))
        return rewards  # .squeeze()

    def generate_from_batch(
        self,
        dataset_name: str,
        records: list[dict[str, Any]],
        original_records: list[SamplingDatasetRecord] | None = None,
    ) -> list[RMSamplingInferenceOutput]:
        self._tokenizer.padding_side = 'left'
        self._tokenizer.pad_token_id = self._tokenizer.pad_token_id

        merged_inputs = [inputs for record in records for key, inputs in record['answers'].items()]

        input_ids = [record['input_ids'].tolist() for record in merged_inputs]
        attention_mask = [record['attention_mask'].tolist() for record in merged_inputs]

        rewards = []
        for i in range(0, len(input_ids), self._micro_batch):
            input_ids_batch = input_ids[i : i + self._micro_batch]
            attn_mask_batch = attention_mask[i : i + self._micro_batch]

            max_input_length = max(len(sample) for sample in input_ids_batch)

            records_batch = self._tokenizer.pad(
                {
                    'input_ids': input_ids_batch,
                    'attention_mask': attn_mask_batch,
                },
                padding='max_length',
                max_length=max_input_length,
                return_tensors='pt',
            ).to(self.device)

            rewards.extend(self.generate_from_batch_records(records_batch).tolist())

        outputs = []
        for i, record in enumerate(original_records):
            record_rewards = {answer.id: rewards[i + j] for j, answer in enumerate(record.answers)}

            outputs.append(
                RMSamplingInferenceOutput(
                    id=record.id,
                    answers=record.answers,
                    dataset_name=record.dataset_name,
                    messages=record.messages,
                    rewards=record_rewards,
                )
            )

        return outputs

class RMSamplingGenerator(BaseGenerator[SamplingDatasetRecord, RMSamplingInferenceOutput]):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, micro_batch: int = 1, **kwargs):
        self._collator = DataCollatorWithPadding(tokenizer=tokenizer)
        self._micro_batch = micro_batch
        super().__init__(tokenizer=tokenizer, **kwargs)

    def generate_from_batch_records(self, records: dict[str, torch.Tensor] | BatchEncoding) -> torch.Tensor:
        with torch.no_grad():
            rewards = self._model(**records).logits.cpu()
        return rewards  # .squeeze()

    def generate_from_batch(
        self,
        dataset_name: str,
        records: list[dict[str, Any]],
        original_records: list[SamplingDatasetRecord] | None = None,
    ) -> list[RMSamplingInferenceOutput]:
        self._tokenizer.padding_side = 'left'
        self._tokenizer.pad_token_id = self._tokenizer.pad_token_id

        merged_inputs = [inputs for record in records for key, inputs in record['answers'].items()]

        input_ids = [record['input_ids'].tolist() for record in merged_inputs]
        attention_mask = [record['attention_mask'].tolist() for record in merged_inputs]

        rewards = []
        for i in range(0, len(input_ids), self._micro_batch):
            input_ids_batch = input_ids[i : i + self._micro_batch]
            attn_mask_batch = attention_mask[i : i + self._micro_batch]

            max_input_length = max(len(sample) for sample in input_ids_batch)

            records_batch = self._tokenizer.pad(
                {
                    'input_ids': input_ids_batch,
                    'attention_mask': attn_mask_batch,
                },
                padding='max_length',
                max_length=max_input_length,
                return_tensors='pt',
            ).to(self.device)

            rewards.extend(self.generate_from_batch_records(records_batch).tolist())

        outputs = []
        for i, record in enumerate(original_records):
            record_rewards = {answer.id: rewards[i + j] for j, answer in enumerate(record.answers)}

            outputs.append(
                RMSamplingInferenceOutput(
                    id=record.id,
                    answers=record.answers,
                    dataset_name=record.dataset_name,
                    messages=record.messages,
                    rewards=record_rewards,
                )
            )

        return outputs
