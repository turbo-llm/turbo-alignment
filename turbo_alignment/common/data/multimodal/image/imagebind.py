from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from turbo_alignment.common.data.multimodal.image.base import BaseImageReader
from turbo_alignment.common.data.multimodal.registry import ImageModalityReaderRegistry
from turbo_alignment.settings.modality import ModalityReader


@ImageModalityReaderRegistry.register(ModalityReader.IMAGEBIND)
class ImageBindImageReader(BaseImageReader):
    def __init__(self):
        self.data_transform = transforms.Compose(
            [
                transforms.Resize((224, 224), interpolation=F.InterpolationMode.BICUBIC),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130358, 0.27577711]
                ),
            ]
        )

    def read(self, path: str) -> torch.Tensor:
        image = cv2.imread(path)
        if image is None:
            raise OSError(f'Image not found: {path}')

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transformed_data = self.data_transform(image=image)

        return torch.Tensor(np.moveaxis(transformed_data['image'], -1, 0))
