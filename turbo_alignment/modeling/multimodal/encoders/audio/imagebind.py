from typing import Optional

import torch

from turbo_alignment.modeling.imagebind import ImageBindSingleton
from turbo_alignment.modeling.imagebind.models import ImageBindSettings
from turbo_alignment.modeling.imagebind.models import (
    ModalityType as ImageBindModalityType,
)
from turbo_alignment.modeling.multimodal.encoders.audio.base import BaseAudioEncoder
from turbo_alignment.modeling.multimodal.encoders.registry import (
    ModalityEncoderRegistry,
)
from turbo_alignment.settings.modality import ModalityEncoderType


@ModalityEncoderRegistry.register(ModalityEncoderType.IMAGEBIND_AUDIO)
class ImageBindAudioModeling(BaseAudioEncoder):
    def __init__(self, imagebind_settings: Optional[ImageBindSettings] = None):
        super().__init__()
        if imagebind_settings is None:
            imagebind_settings = ImageBindSettings()
        self.model_imagebind = ImageBindSingleton(imagebind_settings).get()

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        input_values = {ImageBindModalityType.AUDIO: inputs}
        audio_embeddings = self.model_imagebind(input_values)[ImageBindModalityType.AUDIO]
        audio_embeddings = torch.nn.functional.normalize(audio_embeddings, dim=-1)
        return audio_embeddings

    @property
    def emb_dim(self) -> int:
        return self.model_imagebind.modality_heads[ImageBindModalityType.AUDIO][-1].out_features
