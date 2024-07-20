from enum import Enum
from pathlib import Path

from turbo_alignment.settings.base import ExtraFieldsNotAllowedBaseModel


class ModalityEncoderType(str, Enum):
    IMAGEBIND_AUDIO = 'imagebind_audio'
    IMAGEBIND_IMAGE = 'imagebind_image'
    CLIP = 'clip'
    WHISPER = 'whisper'


class ModalityEncoderSettings(ExtraFieldsNotAllowedBaseModel):
    modality_encoder_type: ModalityEncoderType
    is_pickle: bool = False
    encoder_path: Path


class Modality(str, Enum):
    IMAGE = 'image'
    AUDIO = 'audio'
    TEXT = 'text'


class ModalityReader(str, Enum):
    IMAGEBIND = 'imagebind'
    PICKLE = 'pickle'
    CLIP = 'clip'
    WHISPER = 'whisper'


class ModalityReaderSettings(ExtraFieldsNotAllowedBaseModel):
    reader_type: ModalityReader
    reader_path: Path | None


class ModalityProjectorType(str, Enum):
    LLAVA = 'llava'
    C_ABSTRACTOR = 'c_abstractor'
