from abc import ABC

from turbo_alignment.common.data.multimodal.base import BaseModalityReader


class BaseAudioReader(BaseModalityReader, ABC):
    ...
