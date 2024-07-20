from abc import ABC

from wandb.sdk.lib import RunDisabled
from wandb.sdk.wandb_run import Run

from turbo_alignment.common.logging.weights_and_biases import create_wandb_run
from turbo_alignment.settings.pipelines.train.base import BaseTrainExperimentSettings


class LoggingMixin(ABC):
    @staticmethod
    def _get_wandb_run(experiment_settings: BaseTrainExperimentSettings) -> Run | RunDisabled:
        return create_wandb_run(parameters=experiment_settings.wandb_settings, config=experiment_settings.dict())
