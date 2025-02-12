from pathlib import Path
from typing import Optional

import torch
from transformers import CLIPModel

from turbo_alignment.modeling.multimodal.encoders.image.base import BaseImageEncoder
from turbo_alignment.modeling.multimodal.encoders.registry import (
    ModalityEncoderRegistry,
)
from turbo_alignment.settings.modality import ModalityEncoderType


@ModalityEncoderRegistry.register(ModalityEncoderType.CLIP)
class CLIPImageModeling(BaseImageEncoder):
    def __init__(self, encoder_path: Path, model_clip: Optional[CLIPModel] = None, is_pickle: bool = False):
        super().__init__()
        if model_clip is not None:
            self.model_clip = model_clip
        else:
            self.model_clip = CLIPModel.from_pretrained(encoder_path)
        self.is_pickle = is_pickle

    @staticmethod
    def _get_clip_hidden_states(model_clip: CLIPModel, inputs: torch.Tensor, is_pickle: bool = False) -> torch.Tensor:
        if is_pickle:
            return inputs
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava/modeling_llava.py#L213
        # -2 is default value of vision_feature_layer in llava config
        # [1:] is everything after vit [cls] token
        return model_clip.vision_model(inputs.squeeze(1), output_hidden_states=True).hidden_states[-2][
            :, 1:
        ]  # FIXME: squeeze dimension?

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._get_clip_hidden_states(self.model_clip, inputs, self.is_pickle)

    @property
    def emb_dim(self):
        return self.model_clip.config.vision_config.hidden_size

    @property
    def device(self):
        return self.model_clip.device

    @property
    def n_modality_embs(self) -> int:
        image_size = self.model_clip.config.vision_config.image_size
        dummy_pixel_values = torch.empty(1, 3, image_size, image_size)
        hidden_states = self._get_clip_hidden_states(self.model_clip, dummy_pixel_values, is_pickle=False)
        return hidden_states.shape[1]
