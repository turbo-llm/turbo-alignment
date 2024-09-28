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


@MultiModalProjectorRegistry.register(ModalityProjectorType.LLAVA_WITH_REPLICA)
class LlavaWithTextMultiModalProjector(torch.nn.Module):
    def __init__(self, encoder_hidden_size: int, text_hidden_size: int, n_modality_embs: int):
        super().__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.text_hidden_size = text_hidden_size

        self.encoder_to_text_projection = torch.nn.Linear(encoder_hidden_size, text_hidden_size)
        self.text_to_text_projection = torch.nn.Linear(text_hidden_size, text_hidden_size)

        self.attention = torch.nn.MultiheadAttention(embed_dim=text_hidden_size, num_heads=8)

        self.output_layer = torch.nn.Linear(text_hidden_size, text_hidden_size)

    def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        projected_image = self.encoder_to_text_projection(image_features)
        projected_text = self.text_to_text_projection(text_features)

        permuted_projected_image = projected_image.permute(1, 0, 2)  # [image_patches, batch_size, hidden_dim]
        permuted_projected_text = projected_text.permute(1, 0, 2)  # [textual_tokens, batch_size, hidden_dim]

        _, attention_weights = self.attention(
            query=permuted_projected_text, key=permuted_projected_image, value=permuted_projected_image
        )
        weighted_image_features = attention_weights.mean(1).unsqueeze(-1) * projected_image
        # print(weighted_image_features.shape)
        # exit()
        return weighted_image_features
