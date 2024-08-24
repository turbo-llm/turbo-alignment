from typing import Literal

from peft import PeftType, TaskType

from turbo_alignment.settings.base import ExtraFieldsNotAllowedBaseModel


class BasePeftSettings(ExtraFieldsNotAllowedBaseModel):
    name: PeftType
    task_type: TaskType = TaskType.CAUSAL_LM


class LoraSettings(BasePeftSettings):
    name: Literal[PeftType.LORA] = PeftType.LORA
    r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    bias: str = 'none'
    target_modules: list[str] = ['q_proj', 'v_proj']
    modules_to_save: list[str] | None = None


class PrefixTuningSettings(BasePeftSettings):
    name: Literal[PeftType.PREFIX_TUNING] = PeftType.PREFIX_TUNING
    encoder_hidden_size: int
    prefix_projection: bool


class PromptTuningSettings(BasePeftSettings):
    name: Literal[PeftType.PROMPT_TUNING] = PeftType.PROMPT_TUNING
    num_virtual_tokens: int = 32
    prompt_tuning_init_text: str | None = None


class PTuningSettings(BasePeftSettings):
    name: Literal[PeftType.P_TUNING] = PeftType.P_TUNING
    num_virtual_tokens: int = 32
    encoder_reparameterization_type: str = 'MLP'


PEFT_TYPE = PrefixTuningSettings | LoraSettings | PromptTuningSettings | PTuningSettings
