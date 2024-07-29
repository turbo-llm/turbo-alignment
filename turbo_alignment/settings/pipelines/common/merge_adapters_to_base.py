from pathlib import Path

from turbo_alignment.settings.base import ExtraFieldsNotAllowedBaseModel
from turbo_alignment.settings.model import PreTrainedModelSettings
from turbo_alignment.settings.tf.tokenizer import TokenizerSettings


class MergeAdaptersToBaseModelSettings(ExtraFieldsNotAllowedBaseModel):
    model_settings: PreTrainedModelSettings
    tokenizer_settings: TokenizerSettings

    adapter_path: Path
    save_path: Path

    max_shard_size: str = '400MB'
