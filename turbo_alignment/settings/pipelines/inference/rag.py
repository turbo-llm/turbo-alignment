from pathlib import Path
from typing import Sequence

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


class RAGInferenceExperimentSettings(ExtraFieldsNotAllowedBaseModel):
    inference_settings: Sequence[RAGSingleModelInferenceSettings]

    dataset_settings: ChatMultiDatasetSettings
    save_path: Path
