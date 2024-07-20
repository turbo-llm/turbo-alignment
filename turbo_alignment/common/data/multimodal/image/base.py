from abc import ABC

from turbo_alignment.common.data.multimodal.base import BaseModalityReader


class BaseImageReader(BaseModalityReader, ABC):
    ...
