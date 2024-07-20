from typing import Literal

from turbo_alignment.settings.base import ExtraFieldsNotAllowedBaseModel
from turbo_alignment.settings.cherry_pick import ClassificationCherryPickSettings
from turbo_alignment.settings.datasets.classification import (
    ClassificationMultiDatasetSettings,
)
from turbo_alignment.settings.pipelines.train.base import BaseTrainExperimentSettings
from turbo_alignment.settings.tf.trainer import TrainerSettings


class ClassificationLossSettings(ExtraFieldsNotAllowedBaseModel):
    alpha: Literal['auto'] | list[float] | None
    gamma: float


class ClassificationTrainerSettings(TrainerSettings):
    loss_settings: ClassificationLossSettings


class ClassificationTrainExperimentSettings(BaseTrainExperimentSettings):
    trainer_settings: ClassificationTrainerSettings

    train_dataset_settings: ClassificationMultiDatasetSettings
    val_dataset_settings: ClassificationMultiDatasetSettings

    cherry_pick_settings: ClassificationCherryPickSettings
