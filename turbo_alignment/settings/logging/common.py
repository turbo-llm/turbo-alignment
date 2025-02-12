from enum import Enum

from turbo_alignment.settings.base import ExtraFieldsNotAllowedBaseModel


class LoggingType(str, Enum):
    WANDB = 'wandb'
    CLEARML = 'clearml'


class LoggingSettings(ExtraFieldsNotAllowedBaseModel):
    logging_type: LoggingType
