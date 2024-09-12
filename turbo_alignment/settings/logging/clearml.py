from turbo_alignment.settings.base import ExtraFieldsNotAllowedBaseModel


class ClearMLSettings(ExtraFieldsNotAllowedBaseModel):
    project_name: str
    task_name: str
    tags: list[str] = []

    __name__ = 'ClearMLSettings'

    class Config:
        env_prefix: str = 'CLEARML_'
