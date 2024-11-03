from dataclasses import dataclass
from typing import Any, Literal

from pydantic import field_validator
from transformers import TrainingArguments

from turbo_alignment.constants import TRAINER_LOGS_FOLDER
from turbo_alignment.settings.base import ExtraFieldsNotAllowedBaseModel
from turbo_alignment.settings.cherry_pick import ClassificationCherryPickSettings
from turbo_alignment.settings.datasets.classification import (
    ClassificationMultiDatasetSettings,
)
from turbo_alignment.settings.pipelines.train.base import BaseTrainExperimentSettings


class ClassificationLossSettings(ExtraFieldsNotAllowedBaseModel):
    alpha: Literal['auto'] | list[float] | None
    gamma: float


@dataclass
class ClassificationTrainingArguments(TrainingArguments):
    loss_settings: ClassificationLossSettings = ClassificationLossSettings(
        alpha=[1.0],
        gamma=1.0,
    )


class ClassificationTrainExperimentSettings(BaseTrainExperimentSettings):
    training_arguments: ClassificationTrainingArguments

    train_dataset_settings: ClassificationMultiDatasetSettings
    val_dataset_settings: ClassificationMultiDatasetSettings

    cherry_pick_settings: ClassificationCherryPickSettings

    @field_validator('training_arguments', mode='before')
    def create_training_arguments(cls, values: dict[str, Any]) -> ClassificationTrainingArguments:
        loss_settings = values.pop('loss_settings', {})
        return ClassificationTrainingArguments(
            **values,
            output_dir=TRAINER_LOGS_FOLDER,
            report_to=[],
            remove_unused_columns=False,
            label_names=['labels'],
            loss_settings=ClassificationLossSettings(
                **loss_settings,
            ),
        )
