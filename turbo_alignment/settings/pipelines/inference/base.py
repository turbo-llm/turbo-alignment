from pathlib import Path
from typing import Annotated, Sequence

from pydantic import Field

from turbo_alignment.settings.base import ExtraFieldsNotAllowedBaseModel
from turbo_alignment.settings.datasets.chat import ChatMultiDatasetSettings
from turbo_alignment.settings.datasets.classification import (
    ClassificationMultiDatasetSettings,
)
from turbo_alignment.settings.datasets.multimodal import MultimodalMultiDatasetSettings
from turbo_alignment.settings.model import (
    PreTrainedAdaptersModelSettings,
    PreTrainedModelSettings,
)
from turbo_alignment.settings.tf.tokenizer import TokenizerSettings

INFERENCE_DATASETS_SETTINGS = Annotated[
    ChatMultiDatasetSettings | ClassificationMultiDatasetSettings | MultimodalMultiDatasetSettings,
    Field(discriminator='dataset_type'),
]


class SingleModelInferenceSettings(ExtraFieldsNotAllowedBaseModel):
    model_settings: PreTrainedAdaptersModelSettings | PreTrainedModelSettings
    tokenizer_settings: TokenizerSettings
    use_vllm: bool = False
    batch: int = 1
    micro_batch: int = 1


class InferenceExperimentSettings(ExtraFieldsNotAllowedBaseModel):
    inference_settings: Sequence[SingleModelInferenceSettings]

    dataset_settings: INFERENCE_DATASETS_SETTINGS
    save_path: Path
