from typing import Any

from pydantic import field_validator

from turbo_alignment.constants import TRAINER_LOGS_FOLDER
from turbo_alignment.settings.cherry_pick import ChatCherryPickSettings
from turbo_alignment.settings.datasets.pair_preference import (
    PairPreferenceMultiDatasetSettings,
)
from turbo_alignment.settings.pipelines.train.base import BaseTrainExperimentSettings
from turbo_alignment.trainers.dpo import DPOTrainingArguments


class DPOTrainExperimentSettings(BaseTrainExperimentSettings):
    train_dataset_settings: PairPreferenceMultiDatasetSettings
    val_dataset_settings: PairPreferenceMultiDatasetSettings

    cherry_pick_settings: ChatCherryPickSettings | None = None

    training_arguments: DPOTrainingArguments

    @field_validator('training_arguments', mode='before')
    def create_training_arguments(cls, values: dict[str, Any]) -> DPOTrainingArguments:
        return DPOTrainingArguments(
            **values,
            output_dir=TRAINER_LOGS_FOLDER,
            report_to=[],
            remove_unused_columns=False,
        )
