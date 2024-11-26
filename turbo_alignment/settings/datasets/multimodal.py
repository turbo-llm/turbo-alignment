from pathlib import Path
from typing import Literal

from turbo_alignment.settings.base import ExtraFieldsNotAllowedBaseModel
from turbo_alignment.settings.datasets.base import (
    BaseDatasetSettings,
    DatasetType,
    MultiDatasetSettings,
)
from turbo_alignment.settings.datasets.chat import ChatPromptTemplate
from turbo_alignment.settings.modality import (
    Modality,
    ModalityEncoderSettings,
    ModalityReaderSettings,
)


class MultimodalDatasetSettings(BaseDatasetSettings):
    dataset_type: Literal[DatasetType.MULTIMODAL] = DatasetType.MULTIMODAL
    max_tokens_count: int
    only_answer_loss: bool
    prompt_template: ChatPromptTemplate
    modality_token_mapping: dict[Modality, str]
    modality_reader_settings_mapping: dict[Modality, ModalityReaderSettings | None]
    start_modality_token: str
    end_modality_token: str
    n_modality_embeddings: int
    truncate_top: bool
    random_cut: bool = False
    keep_end: bool | None = None
    only_last_replica_loss: bool = False
    ignore_system_prompt: bool = False


class MultimodalMultiDatasetSettings(MultimodalDatasetSettings, MultiDatasetSettings):
    ...


class MultimodalDatasetProcessingSettings(ExtraFieldsNotAllowedBaseModel):
    reader_settings: ModalityReaderSettings
    encoder_settings: ModalityEncoderSettings
    modality: Modality
    dataset_path: Path
    batch_size: int = 64
    output_file_path: Path
