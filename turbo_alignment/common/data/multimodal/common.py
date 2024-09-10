from pathlib import Path

import h5py
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
    @staticmethod
    def _get_h5_file(path: Path) -> Path:
        return list(path.glob('*.h5'))[0]  # FIXME: What if there is more than one h5 file?

    def read(self, path: str) -> torch.Tensor:
        h5_file = self._get_h5_file(Path(path).parent)
        with h5py.File(h5_file, 'r') as f:
            return torch.tensor(f[Path(path).name])
