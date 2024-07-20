from turbo_alignment.settings.base import ExtraFieldsNotAllowedBaseModel


class ModelTransformersSettings(ExtraFieldsNotAllowedBaseModel):
    load_in_8bit: bool = False
    low_cpu_mem_usage: bool = False
    omit_base_model_save: bool | None = None

    trust_remote_code: bool = False
