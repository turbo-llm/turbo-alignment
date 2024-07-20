from typing import Any

import torch
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase

from turbo_alignment.dataset.classification.models import ClassificationDatasetRecord
from turbo_alignment.generators.base import BaseGenerator
from turbo_alignment.settings.generators.outputs.classification import (
    ClassificationInferenceOutput,
)


class ClassificationGenerator(BaseGenerator[ClassificationDatasetRecord, ClassificationInferenceOutput]):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, **kwargs):
        super().__init__(tokenizer=tokenizer, **kwargs)

        self._collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    def _generate_from_batch(
        self, records: list[dict[str, Any]], original_records: list[ClassificationDatasetRecord], dataset_name: str
    ) -> list[ClassificationInferenceOutput]:
        inputs = [{'input_ids': record['input_ids']} for record in records]

        with torch.no_grad():
            input_ids = self._collator(inputs)['input_ids']
            input_ids = input_ids.to(self.device)
            output_logits = self._model(input_ids).logits
            classes = torch.argmax(output_logits, dim=1)

        return [
            ClassificationInferenceOutput(
                id=record.id,
                messages=record.messages,
                predicted_label=cl.item(),
                dataset_name=dataset_name,
            )
            for record, cl in zip(original_records, classes)
        ]
