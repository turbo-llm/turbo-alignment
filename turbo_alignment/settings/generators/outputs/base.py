from turbo_alignment.settings.base import ExtraFieldsNotAllowedBaseModel


class BaseInferenceOutput(ExtraFieldsNotAllowedBaseModel):
    dataset_name: str
