from abc import ABC

from turbo_alignment.modeling.multimodal.encoders.base import BaseModalityEncoder


class BaseAudioEncoder(BaseModalityEncoder, ABC):
    ...
