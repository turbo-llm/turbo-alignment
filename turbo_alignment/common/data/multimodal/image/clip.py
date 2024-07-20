from pathlib import Path

import cv2
import torch
from transformers import CLIPProcessor

from turbo_alignment.common.data.multimodal.image.base import BaseImageReader
from turbo_alignment.common.data.multimodal.registry import ImageModalityReaderRegistry
from turbo_alignment.settings.modality import ModalityReader


@ImageModalityReaderRegistry.register(ModalityReader.CLIP)
class CLIPImageReader(BaseImageReader):
    def __init__(self, reader_path: Path, processor_clip: CLIPProcessor | None = None):
        if processor_clip:
            self.processor_clip = processor_clip
        else:
            self.processor_clip = CLIPProcessor.from_pretrained(reader_path)

    def read(self, path: str) -> torch.Tensor:
        image = cv2.imread(path)
        if image is None:
            raise OSError(f'Image not found: {path}')

        return self.processor_clip(images=image, return_tensors='pt')['pixel_values']
