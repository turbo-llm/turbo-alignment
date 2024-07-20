import torch

from turbo_alignment.modeling.imagebind import ImageBindSingleton
from turbo_alignment.modeling.imagebind.models import ImageBindSettings
from turbo_alignment.modeling.imagebind.models import (
    ModalityType as ImageBindModalityType,
)
from turbo_alignment.modeling.multimodal.encoders.image.base import BaseImageEncoder
from turbo_alignment.modeling.multimodal.encoders.registry import (
    ModalityEncoderRegistry,
)
from turbo_alignment.settings.modality import ModalityEncoderType


@ModalityEncoderRegistry.register(ModalityEncoderType.IMAGEBIND_IMAGE)
class ImageBindImageModeling(BaseImageEncoder):
    def __init__(self, imagebind_settings: ImageBindSettings | None = None):
        super().__init__()

        if imagebind_settings is None:
            imagebind_settings = ImageBindSettings()
        self.model_imagebind = ImageBindSingleton(imagebind_settings).get()

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        input_values = {
            ImageBindModalityType.VISION: inputs,
        }
        image_embeddings = self.model_imagebind(input_values)[ImageBindModalityType.VISION]
        image_embeddings = torch.nn.functional.normalize(image_embeddings, dim=-1)
        return image_embeddings

    @property
    def emb_dim(self) -> int:
        return self.model_imagebind.modality_heads[ImageBindModalityType.VISION][-1].out_features
