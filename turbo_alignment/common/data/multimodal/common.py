from pathlib import Path

import torch

from turbo_alignment.common.data.multimodal.image.base import BaseImageReader
from turbo_alignment.common.data.multimodal.registry import (
    AudioModalityReaderRegistry,
    ImageModalityReaderRegistry,
)
from turbo_alignment.settings.modality import ModalityReader


@AudioModalityReaderRegistry.register(ModalityReader.PICKLE)
@ImageModalityReaderRegistry.register(ModalityReader.PICKLE)
class FileReader(BaseImageReader):
    def __init__(self, **_kwargs):
        ...

    def read(self, path: str) -> torch.Tensor:
        return torch.load(path + '.image.clip.pt')
