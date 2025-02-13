from typing import Sequence
from pydantic import Field
from turbo_alignment.settings.base import ExtraFieldsNotAllowedBaseModel
from turbo_alignment.settings.generators.chat import CustomChatGenerationSettings
from turbo_alignment.settings.pipelines.inference.base import (
    InferenceExperimentSettings,
    SingleModelInferenceSettings,
)
from turbo_alignment.settings.tf.generation import GeneratorTransformersSettings, VLLMGeneratorSettings
from turbo_alignment.settings.tf.vllm import EngineSettings


class ChatGenerationSettings(ExtraFieldsNotAllowedBaseModel):
    transformers_settings: GeneratorTransformersSettings | VLLMGeneratorSettings
    custom_settings: CustomChatGenerationSettings


class ChatSingleModelInferenceSettings(SingleModelInferenceSettings):
    generation_settings: list[ChatGenerationSettings]
    use_vllm: bool = False
    vllm_engine_settings: EngineSettings = Field(default_factory=EngineSettings)


class ChatInferenceExperimentSettings(InferenceExperimentSettings):
    inference_settings: Sequence[ChatSingleModelInferenceSettings]
