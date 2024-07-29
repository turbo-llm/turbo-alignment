from abc import ABC
from pathlib import Path
from typing import Any, overload

from transformers import PreTrainedTokenizerBase

from turbo_alignment.common.data.io import read_jsonl
from turbo_alignment.common.logging import get_project_logger
from turbo_alignment.dataset.base.base import AlignmentDataset
from turbo_alignment.dataset.chat.chat import TrainChatDataset
from turbo_alignment.dataset.chat.models import ChatDatasetRecord
from turbo_alignment.dataset.classification.models import ClassificationDatasetRecord
from turbo_alignment.dataset.registry import ClassificationDatasetTypeRegistry
from turbo_alignment.settings.datasets.base import (
    DatasetSourceSettings,
    DatasetStrategy,
)
from turbo_alignment.settings.datasets.classification import (
    ClassificationDatasetSettings,
)

logger = get_project_logger()


class ClassificationDataset(AlignmentDataset[ClassificationDatasetRecord], ABC):
    def __init__(
        self,
        source: DatasetSourceSettings,
        settings: ClassificationDatasetSettings,
        tokenizer: PreTrainedTokenizerBase,
    ):
        settings.chat_settings.only_answer_loss = False
        self._chat_dataset = TrainChatDataset(
            source=source,
            settings=settings.chat_settings,
            tokenizer=tokenizer,
            read=False,
        )
        super().__init__(source=source, settings=settings, tokenizer=tokenizer)
        self.settings: ClassificationDatasetSettings = settings

        self._read()

    def _encode(self, records: list[ClassificationDatasetRecord], inference: bool) -> list[dict[str, Any] | None]:
        chat_records: list[ChatDatasetRecord] = [
            ChatDatasetRecord(id=record.id, messages=record.messages) for record in records
        ]

        tokenized_chat_records = self._chat_dataset.convert_records(chat_records)

        output: list[dict[str, Any] | None] = []
        for record, tokenized_record in zip(records, tokenized_chat_records):
            if not tokenized_record:
                continue

            classification_tokens = {k: v.squeeze(0) for k, v in tokenized_record.items() if k != 'labels'}
            classification_tokens['labels'] = record.label

            if inference:
                classification_tokens['id'] = record.id

            output.append(classification_tokens)

        return output

    @staticmethod
    @overload
    def _read_records(records: Path) -> list[ClassificationDatasetRecord]:
        ...

    @staticmethod
    @overload
    def _read_records(records: list[dict]) -> list[ClassificationDatasetRecord]:
        ...

    @staticmethod
    def _read_records(records) -> list[ClassificationDatasetRecord]:
        if isinstance(records, Path):
            return [ClassificationDatasetRecord(**record) for record in read_jsonl(records)]
        if isinstance(records, list):
            return [ClassificationDatasetRecord(**record) for record in records]
        raise NotImplementedError


@ClassificationDatasetTypeRegistry.register(DatasetStrategy.TRAIN)
class TrainClassificationDataset(ClassificationDataset):
    def convert_records(self, records: list[ClassificationDatasetRecord]) -> list[dict[str, Any] | None]:
        return self._encode(records, inference=False)


@ClassificationDatasetTypeRegistry.register(DatasetStrategy.INFERENCE)
class InferenceClassificationDataset(ClassificationDataset):
    def convert_records(self, records: list[ClassificationDatasetRecord]) -> list[dict[str, Any] | None]:
        return self._encode(records, inference=True)
