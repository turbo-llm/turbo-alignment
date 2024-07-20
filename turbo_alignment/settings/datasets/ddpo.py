from typing import Literal

from turbo_alignment.settings.datasets.base import (
    BaseDatasetSettings,
    DatasetType,
    MultiDatasetSettings,
)
from turbo_alignment.settings.datasets.chat import ChatDatasetSettings
from turbo_alignment.settings.datasets.pair_preference import (
    PairPreferenceDatasetSettings,
)


class DDPODatasetSettings(BaseDatasetSettings):
    dataset_type: Literal[DatasetType.DDPO] = DatasetType.DDPO
    chat_settings: ChatDatasetSettings
    pair_preferences: PairPreferenceDatasetSettings


class DDPOMultiDatasetSettings(DDPODatasetSettings, MultiDatasetSettings):
    ...
