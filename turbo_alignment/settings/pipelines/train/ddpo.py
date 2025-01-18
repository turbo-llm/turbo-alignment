from typing import Any

from pydantic import field_validator

from turbo_alignment.constants import TRAINER_LOGS_FOLDER
from turbo_alignment.settings.cherry_pick import ChatCherryPickSettings
from turbo_alignment.settings.datasets.ddpo import DDPOMultiDatasetSettings
from turbo_alignment.settings.model import (
    PreTrainedAdaptersModelSettings,
    PreTrainedModelSettings,
)
from turbo_alignment.settings.pipelines.train.base import BaseTrainExperimentSettings
from turbo_alignment.settings.tf.tokenizer import TokenizerSettings
from turbo_alignment.trainers.ddpo import DDPOTrainingArguments


class DDPOTrainExperimentSettings(BaseTrainExperimentSettings):
    rm_settings: PreTrainedModelSettings | PreTrainedAdaptersModelSettings

    train_dataset_settings: DDPOMultiDatasetSettings
    val_dataset_settings: DDPOMultiDatasetSettings

    rm_tokenizer_settings: TokenizerSettings

    training_arguments: DDPOTrainingArguments

    @field_validator('training_arguments', mode='before')
    def create_training_arguments(cls, values: dict[str, Any]) -> DDPOTrainingArguments:
        return DDPOTrainingArguments(
            **values, output_dir=TRAINER_LOGS_FOLDER, report_to=[], remove_unused_columns=False, label_names=[]
        )
    cherry_pick_settings: ChatCherryPickSettings | None = None
