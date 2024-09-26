import torch

from turbo_alignment.modeling.multimodal.projectors.registry import (
    MultiModalProjectorRegistry,
)
from turbo_alignment.settings.modality import ModalityProjectorType


@MultiModalProjectorRegistry.register(ModalityProjectorType.ATTENTION_POOLING)
class AttentionPoolingMultiModalProjector(torch.nn.Module):
    def __init__(self, encoder_hidden_size: int, text_hidden_size: int, n_modality_embs: int):
        super().__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.text_hidden_size = text_hidden_size
        self.n_modality_embs = n_modality_embs
        self.linear_projection = torch.nn.Linear(encoder_hidden_size, text_hidden_size)
        self.attention_scores = torch.nn.Linear(text_hidden_size, 1)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        projected_features = self.linear_projection(image_features)
        attention_scores = torch.softmax(self.attention_scores(projected_features), 1)
        pooled_output = torch.sum(projected_features * attention_scores, dim=1)
        return pooled_output
