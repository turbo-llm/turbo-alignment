from pathlib import Path
from typing import Any, Sequence

from pydantic import field_validator
from transformers import GenerationConfig

from turbo_alignment.settings.base import ExtraFieldsNotAllowedBaseModel
from turbo_alignment.settings.datasets.chat import ChatMultiDatasetSettings
from turbo_alignment.settings.pipelines.inference.chat import ChatGenerationSettings
from turbo_alignment.settings.rag_model import RAGPreTrainedModelSettings
from turbo_alignment.settings.tf.tokenizer import TokenizerSettings


class RAGSingleModelInferenceSettings(ExtraFieldsNotAllowedBaseModel):
    model_settings: RAGPreTrainedModelSettings
    tokenizer_settings: TokenizerSettings
    batch: int = 1

    generation_settings: list[ChatGenerationSettings]

    @field_validator('generation_settings', mode='before')
    def convert_generation_settings(cls, values: list[Any]) -> GenerationConfig:
        return [ChatGenerationSettings(**value) for value in values]


class RAGInferenceExperimentSettings(ExtraFieldsNotAllowedBaseModel):
    inference_settings: Sequence[RAGSingleModelInferenceSettings]

    dataset_settings: ChatMultiDatasetSettings
    save_path: Path
