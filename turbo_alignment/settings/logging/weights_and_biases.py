from enum import Enum

from pydantic_settings import BaseSettings


class WandbMode(str, Enum):
    ONLINE: str = 'online'
    OFFLINE: str = 'offline'
    DISABLED: str = 'disabled'


class WandbSettings(BaseSettings):
    project_name: str
    run_name: str
    entity: str
    tags: list[str] = []
    notes: str | None = None
    mode: WandbMode = WandbMode.ONLINE

    __name__ = 'WandbSettings'

    class Config:
        env_prefix: str = 'WANDB_'
