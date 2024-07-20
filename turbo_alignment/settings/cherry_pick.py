from turbo_alignment.settings.base import ExtraFieldsNotAllowedBaseModel
from turbo_alignment.settings.datasets.base import MultiDatasetSettings
from turbo_alignment.settings.datasets.chat import ChatMultiDatasetSettings
from turbo_alignment.settings.datasets.classification import (
    ClassificationMultiDatasetSettings,
)
from turbo_alignment.settings.datasets.multimodal import MultimodalMultiDatasetSettings
from turbo_alignment.settings.datasets.pair_preference import (
    PairPreferenceMultiDatasetSettings,
)
from turbo_alignment.settings.generators.chat import CustomChatGenerationSettings
from turbo_alignment.settings.metric import MetricSettings
from turbo_alignment.settings.tf.generation import GeneratorTransformersSettings


class CherryPickSettings(ExtraFieldsNotAllowedBaseModel):
    dataset_settings: MultiDatasetSettings
    metric_settings: list[MetricSettings]


class RMCherryPickSettings(CherryPickSettings):
    dataset_settings: PairPreferenceMultiDatasetSettings


class ClassificationCherryPickSettings(CherryPickSettings):
    dataset_settings: ClassificationMultiDatasetSettings


class GenerationSettings(CherryPickSettings):
    generator_transformers_settings: GeneratorTransformersSettings
    custom_generation_settings: CustomChatGenerationSettings


class ChatCherryPickSettings(GenerationSettings):
    dataset_settings: ChatMultiDatasetSettings


class MultimodalCherryPickSettings(GenerationSettings):
    dataset_settings: MultimodalMultiDatasetSettings
