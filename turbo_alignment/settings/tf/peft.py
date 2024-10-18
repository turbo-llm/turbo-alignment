from typing import Literal

from peft import (
    LoraConfig,
    PeftConfig,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
    PromptTuningConfig,
)

from turbo_alignment.settings.base import ExtraFieldsNotAllowedBaseModel


class BasePeftSettings(ExtraFieldsNotAllowedBaseModel):
    name: PeftType
    config: PeftConfig


class LoraSettings(BasePeftSettings):
    name: Literal[PeftType.LORA] = PeftType.LORA
    config: LoraConfig


class PrefixTuningSettings(BasePeftSettings):
    name: Literal[PeftType.PREFIX_TUNING] = PeftType.PREFIX_TUNING
    config: PrefixTuningConfig


class PromptTuningSettings(BasePeftSettings):
    name: Literal[PeftType.PROMPT_TUNING] = PeftType.PROMPT_TUNING
    config: PromptTuningConfig


class PTuningSettings(BasePeftSettings):
    name: Literal[PeftType.P_TUNING] = PeftType.P_TUNING
    config: PromptEncoderConfig


PEFT_TYPE = PrefixTuningSettings | LoraSettings | PromptTuningSettings | PTuningSettings
