from dataclasses import dataclass
from typing import Any

import torch
import transformers
from transformers import PreTrainedTokenizerBase

from turbo_alignment.constants import DISABLE_LOSS_LABEL


@dataclass
class PairPreferenceDataCollator:
    tokenizer: PreTrainedTokenizerBase
    add_labels: bool = True
    pad_to_multiple_of: int | None = None

    def _get_batch(
        self, examples: list[dict[str, dict[str, Any]]], tokenizer: PreTrainedTokenizerBase, key: str, max_length: int
    ) -> transformers.BatchEncoding:
        features = [ex[key] for ex in examples]
        labels = [v.tolist() for feature in features for k, v in feature.items() if k == 'labels']
        no_labels_features = [{k: v for k, v in feature.items() if k != 'labels'} for feature in features]

        batch = tokenizer.pad(
            no_labels_features,
            padding='max_length',
            max_length=max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )
        if self.add_labels:
            batch['labels'] = torch.tensor(
                [label + (max_length - len(label)) * [DISABLE_LOSS_LABEL] for label in labels]
            )
        return batch

    def __call__(self, examples: list[dict[str, dict[str, Any]]]) -> dict[str, Any]:
        max_length = 0
        for ex in examples:
            for t in ex:
                if 'input_ids' in ex[t]:
                    max_length = max(max_length, len(ex[t]['input_ids']))

        batch = {
            'inputs_w': dict(self._get_batch(examples, self.tokenizer, 'inputs_w', max_length)),
            'inputs_l': dict(self._get_batch(examples, self.tokenizer, 'inputs_l', max_length)),
        }
        if 'best_decode' in examples[0] and len(examples[0]['best_decode']) != 0:
            batch['best_decode'] = dict(self._get_batch(examples, self.tokenizer, 'best_decode', max_length))
        return batch
