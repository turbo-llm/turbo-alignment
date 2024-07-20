from enum import Enum
from pathlib import Path

from pydantic import BaseModel

from turbo_alignment.settings.base import ExtraFieldsNotAllowedBaseModel


class ImageBindModelSettings(ExtraFieldsNotAllowedBaseModel):
    pass


class ModalityType(str, Enum):
    VISION = 'vision'
    TEXT = 'text'
    AUDIO = 'audio'


class ImageBindArchitectureSettings(BaseModel):
    vision_embed_dim: int = 1280
    vision_num_blocks: int = 32
    vision_num_heads: int = 16

    rgbt_kernel_size: tuple[int, int, int] = (2, 14, 14)
    rgbt_stride: tuple[int, int, int] = (2, 14, 14)

    text_embed_dim: int = 1024
    text_num_blocks: int = 24
    text_num_heads: int = 16

    audio_kernel_size: int = 16
    audio_num_blocks: int = 12
    audio_num_heads: int = 12
    audio_embed_dim: int = 768
    audio_drop_path_rate: float = 0.1

    audio_stride: int = 10
    audio_num_mel_bins: int = 128
    audio_target_len: int = 204

    out_embed_dim: int = 1024


class ImageBindSettings(BaseModel):
    architecture_settings: ImageBindArchitectureSettings = ImageBindArchitectureSettings()

    weights_path: Path | None = Path('.checkpoints/imagebind_huge.pth')
    is_trainable: bool = False
