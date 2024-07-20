from typing import Literal

from turbo_alignment.settings.datasets.base import (
    BaseDatasetSettings,
    DatasetType,
    MultiDatasetSettings,
)
from turbo_alignment.settings.datasets.chat import ChatDatasetSettings


class PairPreferenceDatasetSettings(BaseDatasetSettings):
    dataset_type: Literal[DatasetType.PAIR_PREFERENCES] = DatasetType.PAIR_PREFERENCES
    chat_settings: ChatDatasetSettings
    add_labels: bool = True


class PairPreferenceMultiDatasetSettings(PairPreferenceDatasetSettings, MultiDatasetSettings):
    ...
