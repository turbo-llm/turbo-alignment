from abc import ABC, abstractmethod

from turbo_alignment.common.logging.clearml import create_clearml_task
from turbo_alignment.common.logging.weights_and_biases import create_wandb_run
from turbo_alignment.common.registry import Registrable
from turbo_alignment.common.tf.callbacks.logging import (
    ClearMLLoggingCallback,
    LoggingCallback,
    WandbLoggingCallback,
)
from turbo_alignment.settings.logging.clearml import ClearMLSettings
from turbo_alignment.settings.logging.weights_and_biases import WandbSettings
from turbo_alignment.settings.pipelines.train.base import BaseTrainExperimentSettings


class LoggingRegistry(Registrable):
    ...


class LoggingMixin(ABC):
    @staticmethod
    @abstractmethod
    def get_logging_callback(experiment_settings: BaseTrainExperimentSettings) -> LoggingCallback:
        ...


@LoggingRegistry.register(WandbSettings.__name__)
class WandbLoggingMixin(LoggingMixin):
    @staticmethod
    def get_logging_callback(experiment_settings: BaseTrainExperimentSettings) -> WandbLoggingCallback:
        logging_settings: WandbSettings = WandbSettings(**experiment_settings.logging_settings.dict())
        wandb_run = create_wandb_run(parameters=logging_settings, config=experiment_settings.dict())
        return WandbLoggingCallback(wandb_run=wandb_run)


@LoggingRegistry.register(ClearMLSettings.__name__)
class ClearMLLoggingMixin(LoggingMixin):
    @staticmethod
    def get_logging_callback(experiment_settings: BaseTrainExperimentSettings) -> ClearMLLoggingCallback:
        logging_settings: ClearMLSettings = ClearMLSettings(**experiment_settings.logging_settings.dict())
        clearml_task = create_clearml_task(parameters=logging_settings, config=experiment_settings.dict())
        return ClearMLLoggingCallback(clearml_task=clearml_task)
