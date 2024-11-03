from turbo_alignment.settings.logging.common import LoggingSettings, LoggingType


class ClearMLSettings(LoggingSettings):
    logging_type: LoggingType = LoggingType.CLEARML
    project_name: str
    task_name: str
    tags: list[str] = []

    __name__ = 'ClearMLSettings'

    class Config:
        env_prefix: str = 'CLEARML_'
