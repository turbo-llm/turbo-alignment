from turbo_alignment.settings.cherry_pick import ChatCherryPickSettings
from turbo_alignment.settings.datasets.chat import ChatMultiDatasetSettings
from turbo_alignment.settings.pipelines.train.base import BaseTrainExperimentSettings


class SftTrainExperimentSettings(BaseTrainExperimentSettings):
    train_dataset_settings: ChatMultiDatasetSettings
    val_dataset_settings: ChatMultiDatasetSettings

    cherry_pick_settings: ChatCherryPickSettings
