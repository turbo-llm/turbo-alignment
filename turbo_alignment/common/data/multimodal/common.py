from pathlib import Path

import torch
from safetensors import safe_open

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
        self.processed_tensors = None

    @staticmethod
    def _get_safetensors_file(path: Path) -> Path:
        return list(path.glob('*.safetensors'))[0]  # FIXME: What if there is more than one safetensors file?

    def read(self, path: str) -> torch.Tensor:
        safetensors_file = self._get_safetensors_file(Path(path).parent)
        if self.processed_tensors is None:
            self.processed_tensors = safe_open(safetensors_file, framework='pt', device='cpu')
        return self.processed_tensors.get_tensor(Path(path).name)
