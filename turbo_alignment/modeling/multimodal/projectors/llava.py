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


# @MultiModalProjectorRegistry.register(ModalityProjectorType.LLAVA_WITH_REPLICA)
# class LlavaWithTextMultiModalProjector(torch.nn.Module):
#     def __init__(self, encoder_hidden_size: int, text_hidden_size: int, n_modality_embs: int):
#         super().__init__()
#         self.encoder_hidden_size = encoder_hidden_size
#         self.text_hidden_size = text_hidden_size

#         self.encoder_to_text_projection = torch.nn.Linear(encoder_hidden_size, text_hidden_size)
#         self.text_to_text_projection = torch.nn.Linear(text_hidden_size, text_hidden_size)

#         self.attention = torch.nn.MultiheadAttention(embed_dim=text_hidden_size, num_heads=8)

#         self.output_layer = torch.nn.Linear(text_hidden_size, text_hidden_size)

#     def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
#         projected_image = self.encoder_to_text_projection(image_features)
#         projected_text = self.text_to_text_projection(text_features)

#         permuted_projected_image = projected_image.permute(1, 0, 2)  # [image_patches, batch_size, hidden_dim]
#         permuted_projected_text = projected_text.permute(1, 0, 2)  # [textual_tokens, batch_size, hidden_dim]

#         _, attention_weights = self.attention(
#             query=permuted_projected_text, key=permuted_projected_image, value=permuted_projected_image
#         )
#         weighted_image_features = attention_weights.mean(1).unsqueeze(-1) * projected_image
#         # print(weighted_image_features.shape)
#         # exit()
#         return weighted_image_features


# @MultiModalProjectorRegistry.register(ModalityProjectorType.LLAVA_WITH_REPLICA)
# class LlavaWithTextMultiModalProjector(torch.nn.Module):
#     def __init__(self, encoder_hidden_size: int, text_hidden_size: int, n_modality_embs: int):
#         super().__init__()
#         self.encoder_hidden_size = encoder_hidden_size
#         self.text_hidden_size = text_hidden_size
#         self.k = n_modality_embs  # Number of top patches to select
#         self.encoder_to_text_projection = torch.nn.Linear(encoder_hidden_size, text_hidden_size)
#         self.text_to_text_projection = torch.nn.Linear(text_hidden_size, text_hidden_size)
#         self.cross_attention = torch.nn.MultiheadAttention(embed_dim=text_hidden_size, num_heads=8)
#         self.output_layer = torch.nn.Linear(text_hidden_size, text_hidden_size)

#     def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
#         # Project the image features into the text hidden space
#         projected_image = self.encoder_to_text_projection(image_features)
#         projected_text = self.text_to_text_projection(text_features)

#         # Permute dimensions for attention
#         permuted_projected_image = projected_image.permute(1, 0, 2)  # [image_patches, batch_size, hidden_dim]
#         permuted_projected_text = projected_text.permute(1, 0, 2)  # [textual_tokens, batch_size, hidden_dim]

#         # Cross-attention: text tokens attend to image patches
#         _, attention_weights = self.cross_attention(
#             query=permuted_projected_text,  # Text queries attend to image patches
#             key=permuted_projected_image,
#             value=permuted_projected_image
#         )

#         # Average attention weights over text tokens to get importance scores for image patches
#         avg_attention_weights = attention_weights.mean(dim=1)  # [batch_size, image_patches]

#         # Select top-k patches based on attention scores
#         _, topk_indices = torch.topk(avg_attention_weights, self.k, dim=1)  # [batch_size, k]
#         topk_image_patches = projected_image.gather(1, topk_indices.unsqueeze(-1).expand(-1, -1, projected_image.size(-1)))  # [batch_size, k, hidden_dim]

#         # Map the top-k patches into the LM embedding space
#         topk_mapped_patches = self.output_layer(topk_image_patches)  # [batch_size, k, text_hidden_size]
#         return topk_mapped_patches  # Output: [batch_size, k, lm_dim]


# @MultiModalProjectorRegistry.register(ModalityProjectorType.LLAVA_WITH_REPLICA)
# class LlavaWithTextMultiModalProjector(torch.nn.Module):
#     def __init__(self, encoder_hidden_size: int, text_hidden_size: int, n_modality_embs: int):
#         super().__init__()
#         self.encoder_hidden_size = encoder_hidden_size
#         self.text_hidden_size = text_hidden_size
#         self.k = n_modality_embs  # Number of top patches to select
#         self.encoder_to_text_projection = torch.nn.Linear(encoder_hidden_size, text_hidden_size)
#         self.text_to_text_projection = torch.nn.Linear(text_hidden_size, text_hidden_size)
#         self.cross_attention = torch.nn.MultiheadAttention(embed_dim=text_hidden_size, num_heads=8)
#         self.output_layer = torch.nn.Linear(text_hidden_size, text_hidden_size)

#     def forward(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
#         # Project the image features into the text hidden space
#         projected_image = self.encoder_to_text_projection(image_features)
#         projected_text = self.text_to_text_projection(text_features)

#         # Permute dimensions for attention
#         permuted_projected_image = projected_image.permute(1, 0, 2)  # [image_patches, batch_size, hidden_dim]
#         permuted_projected_text = projected_text.permute(1, 0, 2)  # [textual_tokens, batch_size, hidden_dim]

#         # Cross-attention: text tokens attend to image patches
#         attention_values, attention_weights = self.cross_attention(
#             query=permuted_projected_image,  # Text queries attend to image patches
#             key=permuted_projected_text,
#             value=permuted_projected_text
#         )
#         attention_values = attention_values.permute(1, 0, 2)

#         mapped_attentioned_values = self.output_layer(attention_values)
#         return mapped_attentioned_values


@MultiModalProjectorRegistry.register(ModalityProjectorType.LLAVA_WITH_REPLICA)
class LlavaWithTextMultiModalProjector(torch.nn.Module):
    def __init__(self, encoder_hidden_size: int, text_hidden_size: int, n_modality_embs: int):
        super().__init__()
        self.cross_attention = torch.nn.MultiheadAttention(embed_dim=text_hidden_size, num_heads=8)
        self.image_projection = torch.nn.Linear(encoder_hidden_size, text_hidden_size)
        self.text_projection = torch.nn.Linear(text_hidden_size, text_hidden_size)
        self.final_projection = torch.nn.Linear(text_hidden_size, text_hidden_size)
    
    def forward(self, image_features, text_features):
        projected_image = self.image_projection(image_features)
        # projected_text = self.text_projection(text_features)

        image_patches = projected_image.transpose(0, 1)
        # text_tokens = projected_text.transpose(0, 1)
        text_tokens = text_features.transpose(0, 1)

        _, attention_weights = self.cross_attention(query=text_tokens, 
                                                    key=image_patches, 
                                                    value=image_patches)
        
        patch_importance = attention_weights.mean(dim=1)

        attended_image_features = projected_image * patch_importance.unsqueeze(-1)

        # return self.final_projection(attended_image_features.sum(1).unsqueeze(1))
        return attended_image_features.sum(1).unsqueeze(1)
