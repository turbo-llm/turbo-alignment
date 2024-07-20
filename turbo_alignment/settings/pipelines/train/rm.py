from turbo_alignment.settings.cherry_pick import RMCherryPickSettings
from turbo_alignment.settings.datasets.pair_preference import (
    PairPreferenceMultiDatasetSettings,
)
from turbo_alignment.settings.pipelines.train.base import BaseTrainExperimentSettings


class RMTrainExperimentSettings(BaseTrainExperimentSettings):
    train_dataset_settings: PairPreferenceMultiDatasetSettings
    val_dataset_settings: PairPreferenceMultiDatasetSettings

    cherry_pick_settings: RMCherryPickSettings
