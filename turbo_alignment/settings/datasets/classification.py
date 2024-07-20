from typing import Literal

from turbo_alignment.settings.datasets.base import (
    BaseDatasetSettings,
    DatasetType,
    MultiDatasetSettings,
)
from turbo_alignment.settings.datasets.chat import ChatDatasetSettings


class ClassificationDatasetSettings(BaseDatasetSettings):
    dataset_type: Literal[DatasetType.CLASSIFICATION] = DatasetType.CLASSIFICATION
    chat_settings: ChatDatasetSettings


class ClassificationMultiDatasetSettings(ClassificationDatasetSettings, MultiDatasetSettings):
    ...
