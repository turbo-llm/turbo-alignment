from typing import Literal

from turbo_alignment.settings.datasets.base import DatasetType, MultiDatasetSettings
from turbo_alignment.settings.datasets.chat import ChatDatasetSettings


class SamplingDatasetSettings(MultiDatasetSettings):
    dataset_type: Literal[DatasetType.SAMPLING] = DatasetType.SAMPLING
    chat_dataset: ChatDatasetSettings
