from typing import Any

import ray
import torch
from transformers import BatchEncoding, DataCollatorWithPadding, PreTrainedTokenizerBase
from turbo_alignment.common.data.io import read_jsonl
from turbo_alignment.dataset.sampling.models import SamplingDatasetRecord
from turbo_alignment.generators.base import BaseGenerator
from turbo_alignment.settings.generators.outputs.rm import RMSamplingInferenceOutput
from pathlib import Path

class RayRMSamplingGenerator(BaseGenerator[SamplingDatasetRecord, RMSamplingInferenceOutput]):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, micro_batch: int = 1, model_replicas: int = 1, **kwargs):
        self._collator = DataCollatorWithPadding(tokenizer=tokenizer)
        self._micro_batch = micro_batch
        self.model_replicas = model_replicas
        super().__init__(tokenizer=tokenizer, **kwargs)

    def generate_from_batch_records(self, records: dict[str, torch.Tensor] | BatchEncoding) -> torch.Tensor:
        with torch.no_grad():
            rm_records = {k: v.cuda() for k, v in records.items() if k not in ('sample_ids', 'sample_contents')}

            rank = torch.distributed.get_rank()

            rewards = []

            exact_match_goldens = read_jsonl(Path("/app/turbo-alignment/tests/fixtures/datasets/chat/chat_id_x_gold.jsonl"))
            id_x_pred_answer = [{'id': id, 'pred_answer': answer} for id, answer in zip(records['sample_ids'], records['sample_contents'])]
            id_x_golden_answer = [{'id': s['id'], 'gold_answer': s['gold']} for s in exact_match_goldens]

            golden_lookup = {d['id']: d['gold_answer'] for d in id_x_golden_answer}

            merged_list = [
                {'id': d['id'], 'pred_answer': d['pred_answer'], 'gold_answer': golden_lookup.get(d['id'])}
                for d in id_x_pred_answer
            ]
            
            exact_match_mask = torch.tensor([0 if 'exact_match' not in sample['id'] else 1 for sample in merged_list])
            exact_match_rewards = torch.tensor([sample['gold_answer'] == sample['pred_answer'] for sample in merged_list])
            rewards = ray.get(self._model.reward_forward(records=rm_records, index=rank % self.model_replicas))

            exact_match_mask = exact_match_mask.to(rewards.device)
            exact_match_rewards = exact_match_rewards.to(rewards.device)

            new_rewards = torch.where(exact_match_mask == 0, rewards, exact_match_rewards)

        return rewards

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

        reward_index = 0
        record_rewards = []
        for record in records:
            mapped_rewards = {}
            for key in record['answers'].keys():
                mapped_rewards[key] = rewards[reward_index].item()
                reward_index += 1
            record_rewards.append(mapped_rewards)

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