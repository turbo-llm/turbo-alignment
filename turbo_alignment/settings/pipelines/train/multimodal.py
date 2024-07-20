from pathlib import Path

from turbo_alignment.settings.cherry_pick import MultimodalCherryPickSettings
from turbo_alignment.settings.datasets.multimodal import MultimodalMultiDatasetSettings
from turbo_alignment.settings.modality import (
    Modality,
    ModalityEncoderSettings,
    ModalityProjectorType,
)
from turbo_alignment.settings.pipelines.train.base import BaseTrainExperimentSettings


class MultimodalTrainExperimentSettings(BaseTrainExperimentSettings):
    modality_encoder_settings_mapping: dict[Modality, ModalityEncoderSettings | None]
    modality_projector_mapping: dict[Modality, ModalityProjectorType | None]
    modality_projector_initialization_mapping: dict[Modality, Path | None]

    train_dataset_settings: MultimodalMultiDatasetSettings
    val_dataset_settings: MultimodalMultiDatasetSettings

    cherry_pick_settings: MultimodalCherryPickSettings
