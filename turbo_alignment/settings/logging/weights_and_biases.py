import os
from enum import Enum

from turbo_alignment.settings.logging.common import LoggingSettings, LoggingType


class WandbMode(str, Enum):
    ONLINE = 'online'
    OFFLINE = 'offline'
    DISABLED = 'disabled'


class WandbSettings(LoggingSettings):
    logging_type: LoggingType = LoggingType.WANDB
    project_name: str
    run_name: str
    entity: str
    tags: list[str] = []
    notes: str | None = None
    mode: WandbMode = WandbMode.ONLINE

    __name__ = 'WandbSettings'

    def __init__(self, **kwargs) -> None:
        mode_from_env = os.getenv('WANDB_MODE', None)

        if mode_from_env:
            kwargs['mode'] = WandbMode(mode_from_env)

        super().__init__(**kwargs)
