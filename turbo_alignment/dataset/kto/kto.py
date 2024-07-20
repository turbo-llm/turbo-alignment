import random
from copy import deepcopy
from pathlib import Path
from typing import Any, overload

from transformers import PreTrainedTokenizerBase

from turbo_alignment.common.data.io import read_jsonl
from turbo_alignment.common.logging import get_project_logger
from turbo_alignment.dataset.base import AlignmentDataset
from turbo_alignment.dataset.chat import ChatDatasetRecord, TrainChatDataset
from turbo_alignment.dataset.kto.models import KTODatasetRecord
from turbo_alignment.dataset.registry import KTODatasetTypeRegistry
from turbo_alignment.settings.datasets.base import (
    DatasetSourceSettings,
    DatasetStrategy,
)
from turbo_alignment.settings.datasets.kto import KTODatasetSettings

logger = get_project_logger()


@KTODatasetTypeRegistry.register(DatasetStrategy.TRAIN)
class KTODataset(AlignmentDataset[KTODatasetRecord]):
    def __init__(
        self,
        source: DatasetSourceSettings,
        settings: KTODatasetSettings,
        tokenizer: PreTrainedTokenizerBase,
    ):
        self._chat_dataset = TrainChatDataset(
            source=source,
            settings=settings.chat_settings,
            tokenizer=tokenizer,
            read=False,
        )
        super().__init__(source=source, settings=settings, tokenizer=tokenizer)

        self._read()

    def convert_records(self, records: list[KTODatasetRecord]) -> list[dict[str, Any] | None]:
        shuffled_records = deepcopy(records)
        random.shuffle(shuffled_records)

        kl_records = []
        chat_records = []

        for record, shuffled_record in zip(records, shuffled_records):
            chat_records.append(ChatDatasetRecord(id=record.id, messages=record.context + [record.answer]))
            kl_records.append(ChatDatasetRecord(id=record.id, messages=record.context + [shuffled_record.answer]))

        tokenized_chat_records = self._chat_dataset.convert_records(chat_records)
        tokenized_kl_records = self._chat_dataset.convert_records(kl_records)

        output: list[dict[str, Any] | None] = []
        for record, tokenized_record, tokenized_kl_record in zip(
            records, tokenized_chat_records, tokenized_kl_records
        ):
            if not (tokenized_record and tokenized_kl_record):
                continue

            chat_tokens = {k: v.squeeze(0) for k, v in tokenized_record.items()}
            kl_tokens = {k: v.squeeze(0) for k, v in tokenized_kl_record.items()}

            output.append(
                {
                    'is_desirable': record.is_desirable,
                    'chat': chat_tokens,
                    'KL': kl_tokens,
                }
            )

        return output

    @staticmethod
    @overload
    def _read_records(records: Path) -> list[KTODatasetRecord]:
        ...

    @staticmethod
    @overload
    def _read_records(records: list[dict]) -> list[KTODatasetRecord]:
        ...

    @staticmethod
    def _read_records(records) -> list[KTODatasetRecord]:
        if isinstance(records, Path):
            return [KTODatasetRecord(**record) for record in read_jsonl(records)]
        if isinstance(records, list):
            return [KTODatasetRecord(**record) for record in records]
        raise NotImplementedError
