from typing import Sequence

from turbo_alignment.settings.metric import MetricSettings
from turbo_alignment.settings.modality import (
    Modality,
    ModalityEncoderSettings,
    ModalityProjectorType,
)
from turbo_alignment.settings.model import PreTrainedMultiModalModel
from turbo_alignment.settings.pipelines.inference.base import (
    InferenceExperimentSettings,
)
from turbo_alignment.settings.pipelines.inference.chat import (
    ChatSingleModelInferenceSettings,
)


class MultimodalSingleModelInferenceSettings(ChatSingleModelInferenceSettings):
    modality_encoder_settings_mapping: dict[Modality, ModalityEncoderSettings | None]
    modality_projector_mapping: dict[Modality, ModalityProjectorType | None]
    model_settings: PreTrainedMultiModalModel
    metric_settings: list[MetricSettings] | None


class MultimodalInferenceExperimentSettings(InferenceExperimentSettings):
    inference_settings: Sequence[MultimodalSingleModelInferenceSettings]
