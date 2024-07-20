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
    notes: str | None = None
    tags: list[str] = []
    mode: WandbMode = WandbMode.ONLINE

    class Config:
        env_prefix: str = 'WANDB_'
