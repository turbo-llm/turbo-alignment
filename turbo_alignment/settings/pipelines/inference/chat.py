from typing import Sequence

from turbo_alignment.settings.base import ExtraFieldsNotAllowedBaseModel
from turbo_alignment.settings.generators.chat import CustomChatGenerationSettings
from turbo_alignment.settings.pipelines.inference.base import (
    InferenceExperimentSettings,
    SingleModelInferenceSettings,
)
from turbo_alignment.settings.tf.generation import GeneratorTransformersSettings


class ChatGenerationSettings(ExtraFieldsNotAllowedBaseModel):
    transformers_settings: GeneratorTransformersSettings
    custom_settings: CustomChatGenerationSettings


class ChatSingleModelInferenceSettings(SingleModelInferenceSettings):
    generation_settings: list[ChatGenerationSettings]
    use_vllm: bool = False
    tensor_parallel_size: int = 1


class ChatInferenceExperimentSettings(InferenceExperimentSettings):
    inference_settings: Sequence[ChatSingleModelInferenceSettings]
