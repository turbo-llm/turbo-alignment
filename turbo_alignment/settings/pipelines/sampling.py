from pathlib import Path

from turbo_alignment.settings.base import ExtraFieldsNotAllowedBaseModel
from turbo_alignment.settings.datasets.sampling import SamplingDatasetSettings
from turbo_alignment.settings.model import (
    PreTrainedAdaptersModelSettings,
    PreTrainedModelSettings,
)
from turbo_alignment.settings.tf.tokenizer import TokenizerSettings


class BaseSamplingSettings(ExtraFieldsNotAllowedBaseModel):
    dataset_settings: SamplingDatasetSettings

    save_path: Path
    N: int


class RandomSamplingSettings(BaseSamplingSettings):
    ...


class BaseSamplingWithRMSettings(BaseSamplingSettings):
    rm: PreTrainedAdaptersModelSettings | PreTrainedModelSettings
    tokenizer_settings: TokenizerSettings

    rm_batch_size: int


class SamplingWithRMSettings(BaseSamplingWithRMSettings):
    ...


class RSOSamplingSettings(BaseSamplingWithRMSettings):
    beta: float
