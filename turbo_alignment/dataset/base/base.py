import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, TypeVar, overload

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from turbo_alignment.common.logging import get_project_logger
from turbo_alignment.dataset.base.models import DatasetRecord
from turbo_alignment.settings.datasets.base import (
    BaseDatasetSettings,
    DatasetSourceSettings,
)

logger = get_project_logger()


RecordT = TypeVar('RecordT', bound=DatasetRecord)


class BaseDataset(Dataset, ABC, Generic[RecordT]):
    def __init__(
        self,
        source: DatasetSourceSettings,
        settings: BaseDatasetSettings,
    ) -> None:
        self.source = source
        self.settings = settings

        self.original_records_map: dict[str, RecordT] = {}
        self.records: list[dict[str, torch.Tensor]] = []

    def _read(self) -> None:
        if self.source.records_data:
            records = self._read_records(self.source.records_data)
        elif self.source.records_path:
            records = self._read_records(self.source.records_path)
        else:
            raise ValueError('At least one of records_data and records_path should be not None')

        if self.source.offset is not None and self.source.n_rows is not None:
            records = records[self.source.offset : self.source.offset + self.source.n_rows]

        self.original_records_map, self.records = self._sample_dataset(records)

        logger.info(f'Sampled {len(self.records)} records with offset {self.source.offset}')

    def _sample_dataset(
        self,
        original_records: list[RecordT],
    ) -> tuple[dict[str, RecordT], list[dict[str, Any]]]:
        if self.source.sample_rate is not None:
            logger.info(f'Sampling dataset {self.source.name} with sample rate: {self.source.sample_rate}')
            sampled_original_records = {
                record.id: record for record in original_records if random.random() <= self.source.sample_rate
            }
        elif self.source.num_samples is not None:
            logger.info(f'Sampling {self.source.num_samples} from dataset {self.source.name}')
            sampled_original_records = {
                record.id: record
                for record in random.sample(original_records, k=min(self.source.num_samples, len(original_records)))
            }
        else:
            raise ValueError('neither sample_rate nor num_samples are not set')

        sampled_records = [r for r in self.convert_records(list(sampled_original_records.values())) if r is not None]

        return sampled_original_records, sampled_records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]

    def __iter__(self):
        return iter(self.records)

    def get_original_record_by_id(self, record_id: str) -> RecordT:
        return self.original_records_map[record_id]

    @abstractmethod
    def convert_records(self, records: list[RecordT]) -> list[dict[str, Any] | None]:
        ...

    @staticmethod
    @abstractmethod
    @overload
    def _read_records(records: Path) -> list[RecordT]:
        ...

    @staticmethod
    @abstractmethod
    @overload
    def _read_records(records: list[dict]) -> list[RecordT]:
        ...

    @staticmethod
    @abstractmethod
    def _read_records(records):
        ...


class AlignmentDataset(BaseDataset, ABC, Generic[RecordT]):
    def __init__(
        self,
        source: DatasetSourceSettings,
        settings: BaseDatasetSettings,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        super().__init__(source=source, settings=settings)

        self.tokenizer = tokenizer
        self._logged = False

    def _log_example(self, prompt: str, answer: str | None = None) -> None:
        if not self._logged:
            message = f'Source and target examples:\n' f'Prompt: {prompt}\n'
            if answer:
                message += f'Answer: {answer}'
            logger.info(message)
            self._logged = True
