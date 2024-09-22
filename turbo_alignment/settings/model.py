from enum import Enum
from pathlib import Path

from turbo_alignment.settings.base import ExtraFieldsNotAllowedBaseModel
from turbo_alignment.settings.tf.model import ModelTransformersSettings
from turbo_alignment.settings.tf.peft import PEFT_TYPE


class ModelType(str, Enum):
    CAUSAL = 'causal'
    SEQ2SEQ = 'seq2seq'
    SEQ_CLS = 'seq_cls'
    ENC = 'encoder'


class PreTrainedModelSettings(ExtraFieldsNotAllowedBaseModel):
    model_path: Path
    model_type: ModelType

    model_kwargs: dict = {}

    transformers_settings: ModelTransformersSettings

    embeddings_initialization_strategy: dict[str, str] | None = None

    use_liger_kernels: bool = False


class PreTrainedAdaptersModelSettings(PreTrainedModelSettings):
    adapter_path: Path
    is_trainable: bool = False


class ModelForPeftSettings(PreTrainedModelSettings):
    peft_settings: PEFT_TYPE


class PreTrainedMultiModalModel(PreTrainedAdaptersModelSettings):
    projections_path: Path
    n_modality_embeddings: int
