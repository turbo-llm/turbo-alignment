from enum import Enum
from pathlib import Path

from pydantic import model_validator

from turbo_alignment.settings.base import ExtraFieldsNotAllowedBaseModel


class DatasetType(str, Enum):
    CHAT = 'chat'
    PAIR_PREFERENCES = 'pair_preferences'
    SAMPLING = 'sampling'
    DDPO = 'ddpo'
    CLASSIFICATION = 'classification'
    MULTIMODAL = 'multimodal'
    KTO = 'kto'


class DatasetStrategy(str, Enum):
    TRAIN = 'train'
    INFERENCE = 'inference'


class BaseDatasetSettings(ExtraFieldsNotAllowedBaseModel):
    dataset_type: DatasetType


class DatasetSourceSettings(ExtraFieldsNotAllowedBaseModel):
    name: str
    system_prompt: str | None = None
    sample_rate: float | None = None
    num_samples: int | None = None
    records_path: Path | None = None
    records_data: list | None = None
    offset: int | None = None
    n_rows: int | None = None

    @model_validator(mode='after')
    def correct_dataset_sampling_values(self) -> 'DatasetSourceSettings':
        if self.sample_rate is None and self.num_samples is None:
            raise ValueError('neither sample_rate nor num_samples are not set')
        if self.sample_rate is not None and self.num_samples is not None:
            raise ValueError('both sample_rate and num_samples are set')
        if self.offset is not None and self.n_rows is None:
            raise ValueError('both offset and num_rows should be set')
        if self.n_rows is not None and self.offset is None:
            raise ValueError('both offset and num_rows should be set')

        return self


class MultiDatasetSettings(ExtraFieldsNotAllowedBaseModel):
    sources: list[DatasetSourceSettings]
    dataset_type: DatasetType
