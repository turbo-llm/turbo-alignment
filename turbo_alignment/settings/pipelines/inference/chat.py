from typing import Any, Sequence

from pydantic import field_validator
from transformers import GenerationConfig

from turbo_alignment.settings.base import ExtraFieldsNotAllowedBaseModel
from turbo_alignment.settings.generators.chat import CustomChatGenerationSettings
from turbo_alignment.settings.pipelines.inference.base import (
    InferenceExperimentSettings,
    SingleModelInferenceSettings,
)


class ChatGenerationSettings(ExtraFieldsNotAllowedBaseModel):
    generation_config: GenerationConfig
    custom_settings: CustomChatGenerationSettings

    @field_validator('generation_config', mode='before')
    def convert_generation_config(cls, values: dict[str, Any]) -> GenerationConfig:
        return GenerationConfig.from_dict(values)


class ChatSingleModelInferenceSettings(SingleModelInferenceSettings):
    generation_settings: list[ChatGenerationSettings]
    use_vllm: bool = False
    tensor_parallel_size: int = 1

    @field_validator('generation_settings', mode='before')
    def convert_generation_settings(cls, values: list[Any]) -> GenerationConfig:
        return [ChatGenerationSettings(**value) for value in values]


class ChatInferenceExperimentSettings(InferenceExperimentSettings):
    inference_settings: Sequence[ChatSingleModelInferenceSettings]
