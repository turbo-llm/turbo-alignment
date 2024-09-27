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


@MultiModalProjectorRegistry.register(ModalityProjectorType.TOP_K_ATTENTION_POOLING)
class TopKAttentionPoolingMultiModalProjector(torch.nn.Module):
    def __init__(self, encoder_hidden_size: int, text_hidden_size: int, n_modality_embs: int):
        super().__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.text_hidden_size = text_hidden_size
        self.n_modality_embs = n_modality_embs
        self.linear_projection = torch.nn.Linear(encoder_hidden_size, text_hidden_size)
        self.attention_scores = torch.nn.Linear(text_hidden_size, 1)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        projected_features = self.linear_projection(
            image_features
        )  # map each image patch to the language model dimension
        attention_scores = torch.softmax(
            self.attention_scores(projected_features), 1
        )  # calculate learnable attention scores for each patch
        top_indices = torch.topk(
            attention_scores.squeeze(-1), k=self.n_modality_embs, dim=1
        ).indices  # select indices top N patches according to attention scores
        top_k_hidden_states = torch.gather(
            projected_features, index=top_indices.unsqueeze(-1).expand(-1, -1, projected_features.size(-1)), dim=1
        )  # select top patches
        return top_k_hidden_states


@MultiModalProjectorRegistry.register(ModalityProjectorType.THRESHOLD_SELECTOR)
class ThresholdSelectorMultiModalProjector(torch.nn.Module):
    def __init__(self, encoder_hidden_size: int, text_hidden_size: int, n_modality_embs: int):
        super().__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.text_hidden_size = text_hidden_size
        self.n_modality_embs = n_modality_embs
        self.linear_projection = torch.nn.Linear(encoder_hidden_size, text_hidden_size)
        self.selection_score = torch.nn.Linear(text_hidden_size, 1)
        self.threshold = 0.5

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        projected_features = self.linear_projection(
            image_features
        )  # map each image patch to the language model dimension
        selection_scores = torch.sigmoid(
            self.selection_score(projected_features)
        )  # calculate learnable attention scores for each patch
        selection_mask = selection_scores < self.threshold
        projected_features[
            :, selection_mask[0, :, 0]
        ] = 0  # set zeros for hiddens with attention score < threshold (just a test for PoC)
        return projected_features
