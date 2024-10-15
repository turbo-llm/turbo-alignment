import math

import numpy as np
import torch
import torch.nn.functional as F

from turbo_alignment.modeling.multimodal.projectors.registry import (
    MultiModalProjectorRegistry,
)
from turbo_alignment.settings.modality import ModalityProjectorType


def get_abs_pos(abs_pos, tgt_size):
    # abs_pos: L, C
    # tgt_size: M
    # return: M, C
    src_size = int(math.sqrt(abs_pos.size(0)))
    tgt_size = int(math.sqrt(tgt_size))
    dtype = abs_pos.dtype

    if src_size != tgt_size:
        return (
            F.interpolate(
                abs_pos.float().reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2),
                size=(tgt_size, tgt_size),
                mode='bicubic',
                align_corners=False,
            )
            .permute(0, 2, 3, 1)
            .flatten(0, 2)
            .to(dtype=dtype)
        )
    else:
        return abs_pos


# https://huggingface.co/Qwen/Qwen-VL-Chat/blob/main/visual.py
# https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/pos_embed.py#L20
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


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
        self.top_k = 128
        self.pos_embed = torch.nn.Parameter(
            torch.from_numpy(get_2d_sincos_pos_embed(text_hidden_size, 24)).float()
        ).requires_grad_(False)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        pos_embed = get_abs_pos(self.pos_embed, image_features.size(1))

        projected_features = self.linear_projection(
            image_features
        )  # map each image patch to the language model dimension

        projected_features = projected_features + pos_embed

        attention_scores = torch.softmax(
            self.attention_scores(projected_features), 1
        )  # calculate learnable attention scores for each patch
        top_indices = torch.topk(
            attention_scores.squeeze(-1), k=attention_scores.shape[1], dim=1
        ).indices  # select indices top N patches according to attention scores

        projected_features[:, top_indices[:, self.top_k :].squeeze(0)] = 0  # set zero for unselected tokens
        projected_features = projected_features[(projected_features != 0).any(dim=-1)]  # remove zero vectors

        return projected_features.unsqueeze(0)


@MultiModalProjectorRegistry.register(ModalityProjectorType.TOP_K_ATTENTION_POOLING_WITH_N_HEADS)
class TopKAttentionPoolingWithNHeadsMultiModalProjector(torch.nn.Module):
    def __init__(self, encoder_hidden_size: int, text_hidden_size: int, n_modality_embs: int):
        super().__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.text_hidden_size = text_hidden_size
        self.n_modality_embs = n_modality_embs
        self.linear_projection = torch.nn.Linear(encoder_hidden_size, text_hidden_size)
        self.num_heads = 1
        self.attention_scores = torch.nn.Linear(text_hidden_size, self.num_heads)
        self.top_k = n_modality_embs
        # self.pos_embed = torch.nn.Parameter(
        #     torch.from_numpy(get_2d_sincos_pos_embed(text_hidden_size, 15)).float()
        # ).requires_grad_(False)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        # pos_embed = get_abs_pos(self.pos_embed, image_features.size(1))

        projected_features = self.linear_projection(
            image_features
        )  # map each image patch to the language model dimension

        # projected_features = projected_features + pos_embed

        scores = self.attention_scores(projected_features)
        attention_scores = torch.softmax(scores, 1)  # calculate learnable attention scores for each patch
        attention_scores = torch.max(attention_scores, -1).values
        # attention_scores = torch.mean(attention_scores, -1)
        top_indices = torch.topk(
            attention_scores.squeeze(-1), k=attention_scores.shape[1], dim=1
        ).indices  # select indices top N patches according to attention scores

        projected_features[:, top_indices[:, self.top_k :].squeeze(0)] = 0  # set zero for unselected tokens
        projected_features = projected_features[(projected_features != 0).any(dim=-1)]  # remove zero vectors

        return projected_features.unsqueeze(0)


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
