from typing import Literal

from turbo_alignment.settings.datasets.base import (
    BaseDatasetSettings,
    DatasetType,
    MultiDatasetSettings,
)
from turbo_alignment.settings.datasets.chat import ChatDatasetSettings


class KTODatasetSettings(BaseDatasetSettings):
    dataset_type: Literal[DatasetType.KTO] = DatasetType.KTO
    chat_settings: ChatDatasetSettings
    shuffle_seed: int = 54


class KTOMultiDatasetSettings(KTODatasetSettings, MultiDatasetSettings):
    ...
