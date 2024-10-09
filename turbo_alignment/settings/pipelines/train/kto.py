from turbo_alignment.settings.cherry_pick import ChatCherryPickSettings
from turbo_alignment.settings.datasets.kto import KTOMultiDatasetSettings
from turbo_alignment.settings.pipelines.train.base import BaseTrainExperimentSettings
from turbo_alignment.trainers.kto import KTOTrainingArguments


class KTOTrainExperimentSettings(BaseTrainExperimentSettings):
    train_dataset_settings: KTOMultiDatasetSettings
    val_dataset_settings: KTOMultiDatasetSettings

    cherry_pick_settings: ChatCherryPickSettings

    # training_arguments: KTOTrainingArguments
