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
        self.processed_batches = None

    @staticmethod
    def _get_pt_files(path: Path) -> Path:
        return list(path.glob('*.pt'))

    def read(self, path: str) -> torch.Tensor:
        print('Calling a read!')
        if self.processed_batches is None:
            self.processed_batches = {}
            pt_files = self._get_pt_files(Path(path).parent)
            for pt_file in pt_files:
                file_tensor_dict = torch.load(pt_file)
                self.processed_batches.update(file_tensor_dict)
        return self.processed_batches[Path(path).name]
