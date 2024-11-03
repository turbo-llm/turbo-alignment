from enum import Enum

from turbo_alignment.settings.base import ExtraFieldsNotAllowedBaseModel


class LoggingType(str, Enum):
    WANDB: str = 'wandb'
    CLEARML: str = 'clearml'


class LoggingSettings(ExtraFieldsNotAllowedBaseModel):
    logging_type: LoggingType
