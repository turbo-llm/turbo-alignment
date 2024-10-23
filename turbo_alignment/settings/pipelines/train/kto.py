from typing import Any

from pydantic import field_validator

from turbo_alignment.constants import TRAINER_LOGS_FOLDER
from turbo_alignment.settings.cherry_pick import ChatCherryPickSettings
from turbo_alignment.settings.datasets.kto import KTOMultiDatasetSettings
from turbo_alignment.settings.pipelines.train.base import BaseTrainExperimentSettings
from turbo_alignment.trainers.kto import KTOTrainingArguments


class KTOTrainExperimentSettings(BaseTrainExperimentSettings):
    train_dataset_settings: KTOMultiDatasetSettings
    val_dataset_settings: KTOMultiDatasetSettings

    cherry_pick_settings: ChatCherryPickSettings

    training_arguments: KTOTrainingArguments

    @field_validator('training_arguments', mode='before')
    def create_training_arguments(cls, values: dict[str, Any]) -> KTOTrainingArguments:
        return KTOTrainingArguments(
            **values, output_dir=TRAINER_LOGS_FOLDER, report_to=[], label_names=[], remove_unused_columns=False
        )
