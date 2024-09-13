from dataclasses import dataclass
from typing import Any

import torch
import transformers
from src.constants import DISABLE_LOSS_LABEL
from transformers import PreTrainedTokenizerBase


@dataclass
class REINFORCEDataCollator:
    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: int | None = None

    def _get_batch(
        self,
        examples: list[dict[str, dict[str, Any]]],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
    ) -> transformers.BatchEncoding:
        features = [ex for ex in examples]
        labels = [
            v.tolist()
            for feature in features
            for k, v in feature.items()
            if k == "labels"
        ]
        no_labels_features = [
            {
                k: v
                for k, v in feature.items()
                if k
                not in ["prompt", "id", "messages", "meta", "labels", "response_mask"]
            }
            for feature in features
        ]

        # print('🤫'*5)
        # print(no_labels_features)

        batch = tokenizer.pad(
            no_labels_features,
            padding="max_length",
            max_length=max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if "response_mask" in examples[0].keys():
            padded_response_masks = []
            for feat in features:
                x = feat["response_mask"]
                padded_response_mask = torch.hstack(
                    [
                        torch.zeros(max_length - len(x)),
                        x,
                    ]
                )
                padded_response_masks.append(padded_response_mask)

            batch["response_mask"] = torch.vstack(padded_response_masks).to(x.dtype)

        batch["labels"] = torch.tensor(
            [
                label + (max_length - len(label)) * [DISABLE_LOSS_LABEL]
                for label in labels
            ]
        )

        # print('👀'*5)
        # print(batch)
        return batch

    def __call__(self, examples: list[dict[str, dict[str, Any]]]) -> dict[str, Any]:
        max_length = 0
        for ex in examples:
            max_length = max(max_length, len(ex["input_ids"]))

        return dict(self._get_batch(examples, self.tokenizer, max_length))
