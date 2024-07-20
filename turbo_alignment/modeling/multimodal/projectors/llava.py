import torch
from transformers.activations import GELUActivation

from turbo_alignment.modeling.multimodal.projectors.registry import (
    MultiModalProjectorRegistry,
)
from turbo_alignment.settings.modality import ModalityProjectorType


@MultiModalProjectorRegistry.register(ModalityProjectorType.LLAVA)
class LlavaMultiModalProjector(torch.nn.Module):
    def __init__(self, encoder_hidden_size: int, text_hidden_size: int, n_modality_embs: int):
        super().__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.text_hidden_size = text_hidden_size
        self.linear_1 = torch.nn.Linear(encoder_hidden_size, text_hidden_size, bias=True)
        self.act = GELUActivation()
        self.linear_2 = torch.nn.Linear(text_hidden_size, text_hidden_size, bias=True)
        self.n_modality_embs = n_modality_embs

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states
