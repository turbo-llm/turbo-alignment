from typing import Any

import torch
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase

from turbo_alignment.dataset.pair_preferences import PairPreferenceRecord
from turbo_alignment.generators.base import BaseGenerator
from turbo_alignment.settings.generators.outputs.rm import RMPairInferenceOutput


class MultiHeadPairGenerator(BaseGenerator[PairPreferenceRecord, RMPairInferenceOutput]):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, **kwargs):
        self._collator = DataCollatorWithPadding(tokenizer=tokenizer)

        super().__init__(tokenizer=tokenizer, **kwargs)

    def _generate_from_batch(
        self, records: list[dict[str, Any]], original_records: list[PairPreferenceRecord], dataset_name: str
    ) -> list[RMPairInferenceOutput]:
        rewards_w, rewards_l = [], []
        for record in records:
            input_ids_w = record['inputs_w']['input_ids'].to(self.device)
            att_w = record['inputs_w']['attention_mask'].to(self.device)
            inputs_w = {'input_ids': input_ids_w, 'attention_mask': att_w}

            input_ids_l = record['inputs_l']['input_ids'].to(self.device)
            att_l = record['inputs_w']['attention_mask'].to(self.device)
            inputs_l = {'input_ids': input_ids_l, 'attention_mask': att_l}

            batch = {'inputs_w': inputs_w, 'inputs_l': inputs_l}

            with torch.no_grad():
                _, reward_w, reward_l, _ = self._model.forward(batch).logits.cpu()
                rewards_w.append(reward_w)
                rewards_l.append(reward_l)

        # merged_inputs = [r['inputs_w'] for r in records] + [r['inputs_l'] for r in records]
        # batch = self._collator(merged_inputs)
        # input_ids = batch['input_ids'].to(self.device)
        # attn_mask = batch['attention_mask'].to(self.device)

        # with torch.no_grad():
        #     _, rewards_w, rewards_l, _ = self._model(input_ids=input_ids, attention_mask=attn_mask).logits.cpu()

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
