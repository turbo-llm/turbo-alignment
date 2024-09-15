from pydantic_settings import BaseSettings


class ClearMLSettings(BaseSettings):
    project_name: str
    task_name: str
    tags: list[str] = []

    __name__ = 'ClearMLSettings'

    class Config:
        env_prefix: str = 'CLEARML_'
