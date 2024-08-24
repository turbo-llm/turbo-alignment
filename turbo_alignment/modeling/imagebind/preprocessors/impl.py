import math
from abc import ABC
from typing import Callable, Literal

import numpy as np
import torch
from timm.models.layers import trunc_normal_

from turbo_alignment.modeling.common.helpers import VerboseNNModule, cast_if_src_dtype


def _get_sinusoid_encoding_table(n_position: int, d_hid: int) -> torch.Tensor:
    """Sinusoid position encoding table"""

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


def _interpolate_pos_encoding_2d(target_spatial_size: int, pos_embed: torch.Tensor) -> torch.Tensor:
    N = pos_embed.shape[1]
    if N == target_spatial_size:
        return pos_embed
    dim = pos_embed.shape[-1]
    # torch.nn.functional.interpolate doesn't work with bfloat16, so we cast to float32
    pos_embed, updated = cast_if_src_dtype(pos_embed, torch.bfloat16, torch.float32)
    pos_embed = torch.nn.functional.interpolate(
        pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
        scale_factor=math.sqrt(target_spatial_size / N),
        mode='bicubic',
    )
    if updated:
        pos_embed, _ = cast_if_src_dtype(pos_embed, torch.float32, torch.bfloat16)
    pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return pos_embed


def _interpolate_pos_encoding(
    npatch_per_img: int,
    pos_embed: torch.Tensor,
    patches_layout: list,
    input_shape: tuple | None = None,
    first_patch_idx: int = 1,
):
    assert first_patch_idx in (0, 1), 'there is 1 CLS token or none'
    N = pos_embed.shape[1] - first_patch_idx  # since it's 1 if cls_token exists
    if npatch_per_img == N:
        return pos_embed

    assert patches_layout[-1] == patches_layout[-2], 'Interpolation of pos embed not supported for non-square layouts'

    class_emb = pos_embed[:, :first_patch_idx]
    pos_embed = pos_embed[:, first_patch_idx:]

    if input_shape is None or patches_layout[0] == 1:
        # simple 2D pos embedding, no temporal component
        pos_embed = _interpolate_pos_encoding_2d(npatch_per_img, pos_embed)
    elif patches_layout[0] > 1:
        # pos embed has a temporal component
        assert len(input_shape) == 4, 'temporal interpolation not supported'
        # we only support 2D interpolation in this case
        num_frames = patches_layout[0]
        num_spatial_tokens = patches_layout[1] * patches_layout[2]
        pos_embed = pos_embed.view(1, num_frames, num_spatial_tokens, -1)
        # interpolate embedding for zeroth frame
        pos_embed = _interpolate_pos_encoding_2d(npatch_per_img, pos_embed[0, 0, ...].unsqueeze(0))
    else:
        raise ValueError("This type of interpolation isn't implemented")

    return torch.cat((class_emb, pos_embed), dim=1)


def _build_causal_attention_mask(context_length: int) -> torch.Tensor:
    # lazily create causal attention mask, with full attention between the vision tokens
    # pytorch uses additive attention mask; fill with -inf
    mask = torch.empty(context_length, context_length, requires_grad=False)
    mask.fill_(float('-inf'))
    mask.triu_(1)  # zero out the lower diagonal
    return mask


def _get_pos_embedding(
    npatch_per_img: int,
    pos_embed: torch.Tensor,
    patches_layout: list,
    input_shape: tuple,
    first_patch_idx=1,
):
    pos_embed = _interpolate_pos_encoding(
        npatch_per_img,
        pos_embed,
        patches_layout,
        input_shape=input_shape,
        first_patch_idx=first_patch_idx,
    )
    return pos_embed


class PatchEmbedGeneric(torch.nn.Module):
    """
    PatchEmbed from Hydra
    """

    def __init__(self, proj_stem: list[torch.nn.Module], norm_layer: torch.nn.Module | None = None):
        super().__init__()

        self.proj: torch.nn.Module
        if len(proj_stem) > 1:
            self.proj = torch.nn.Sequential(*proj_stem)
        else:
            # Special case to be able to load pre-trained models that were
            # trained with a standard stem
            self.proj = proj_stem[0]

        self.norm_layer = norm_layer

    def get_patch_layout(self, img_size):
        with torch.no_grad():
            dummy_img = torch.zeros(
                [
                    1,
                ]
                + img_size
            )
            dummy_out = self.proj(dummy_img)
        embed_dim = dummy_out.shape[1]
        patches_layout = tuple(dummy_out.shape[2:])
        num_patches = np.prod(patches_layout)
        return patches_layout, num_patches, embed_dim

    def forward(self, x):
        with torch.no_grad():
            x = self.proj(x)
            # B C (T) H W -> B (T)HW C
            x = x.flatten(2).transpose(1, 2)
            if self.norm_layer is not None:
                x = self.norm_layer(x)
        return x


class SpatioTemporalPosEmbeddingHelper(VerboseNNModule, ABC):
    def __init__(
        self,
        patches_layout: list,
        num_patches: int,
        num_cls_tokens: int,
        embed_dim: int,
        learnable: bool,
    ) -> None:
        super().__init__()
        self.num_cls_tokens = num_cls_tokens
        self.patches_layout = patches_layout
        self.num_patches = num_patches
        self.num_tokens = num_cls_tokens + num_patches
        self.learnable = learnable
        if self.learnable:
            self.pos_embed = torch.nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
            trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.register_buffer('pos_embed', _get_sinusoid_encoding_table(self.num_tokens, embed_dim))

    def get_pos_embedding(self, vision_input, all_vision_tokens):
        input_shape = vision_input.shape
        pos_embed = _get_pos_embedding(
            all_vision_tokens.size(1) - self.num_cls_tokens,
            pos_embed=self.pos_embed,
            patches_layout=self.patches_layout,
            input_shape=input_shape,
            first_patch_idx=self.num_cls_tokens,
        )
        return pos_embed


class RGBDTPreprocessor(VerboseNNModule):
    def __init__(
        self,
        rgbt_stem: PatchEmbedGeneric | None,
        depth_stem: PatchEmbedGeneric | None,
        img_size: list | None = None,
        num_cls_tokens: int = 1,
        pos_embed_fn: Callable | None = None,
        use_type_embed: bool = False,
        init_param_style: str = 'openclip',
    ) -> None:
        super().__init__()
        stem = rgbt_stem if rgbt_stem is not None else depth_stem
        if img_size is None:
            img_size = [3, 224, 224]
        assert stem is not None
        (
            self.patches_layout,
            self.num_patches,
            self.embed_dim,
        ) = stem.get_patch_layout(img_size)
        self.rgbt_stem = rgbt_stem
        self.depth_stem = depth_stem
        self.use_pos_embed = pos_embed_fn is not None
        self.use_type_embed = use_type_embed
        self.num_cls_tokens = num_cls_tokens

        if pos_embed_fn is not None:
            self.pos_embedding_helper = pos_embed_fn(
                patches_layout=self.patches_layout,
                num_cls_tokens=num_cls_tokens,
                num_patches=self.num_patches,
                embed_dim=self.embed_dim,
            )
        if self.num_cls_tokens > 0:
            self.cls_token = torch.nn.Parameter(torch.zeros(1, self.num_cls_tokens, self.embed_dim))
        if self.use_type_embed:
            self.type_embed = torch.nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        self.init_parameters(init_param_style)

    @torch.no_grad()
    def init_parameters(self, init_param_style):
        if init_param_style == 'openclip':
            # OpenCLIP style initialization
            scale = self.embed_dim**-0.5
            if self.use_pos_embed:
                torch.nn.init.normal_(self.pos_embedding_helper.pos_embed)
                self.pos_embedding_helper.pos_embed *= scale

            if self.num_cls_tokens > 0:
                torch.nn.init.normal_(self.cls_token)
                self.cls_token *= scale
        elif init_param_style == 'vit':
            self.cls_token.data.fill_(0)
        else:
            raise ValueError(f'Unknown init {init_param_style}')

        if self.use_type_embed:
            torch.nn.init.normal_(self.type_embed)

    def tokenize_input_and_cls_pos(self, inputs, stem):
        # tokens is of shape B x L x D
        tokens = stem(inputs)
        assert tokens.ndim == 3
        assert tokens.shape[2] == self.embed_dim
        B = tokens.shape[0]
        if self.num_cls_tokens > 0:
            class_tokens = self.cls_token.expand(B, -1, -1)  # stole class_tokens impl from Phil Wang, thanks
            tokens = torch.cat((class_tokens, tokens), dim=1)
        if self.use_pos_embed:
            pos_embed = self.pos_embedding_helper.get_pos_embedding(inputs, tokens)
            tokens = tokens + pos_embed
        if self.use_type_embed:
            tokens = tokens + self.type_embed.expand(B, -1, -1)
        return tokens

    def forward(self, vision: torch.Tensor | None = None, depth: torch.Tensor | None = None, patch_mask=None) -> dict:
        if patch_mask is not None:
            raise NotImplementedError()

        if vision is not None:
            vision_tokens = self.tokenize_input_and_cls_pos(vision, self.rgbt_stem)

        if depth is not None:
            depth_tokens = self.tokenize_input_and_cls_pos(depth, self.depth_stem)

        # aggregate tokens
        if vision is not None and depth is not None:
            final_tokens = vision_tokens + depth_tokens
        else:
            final_tokens = vision_tokens if vision is not None else depth_tokens
        return_dict = {
            'trunk': {
                'tokens': final_tokens,
            },
            'head': {},
        }
        return return_dict


class AudioPreprocessor(RGBDTPreprocessor):
    def __init__(self, audio_stem: PatchEmbedGeneric, **kwargs) -> None:
        super().__init__(rgbt_stem=audio_stem, depth_stem=None, **kwargs)

    def forward(self, *_args, audio: torch.Tensor | None = None, **_kwargs) -> dict:
        # vision here is actually audio
        return super().forward(vision=audio)


class TextPreprocessor(VerboseNNModule):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        embed_dim: int,
        causal_masking: bool,
        supply_seq_len_to_head: bool = True,
        num_cls_tokens: int = 0,
        init_param_style: str = 'openclip',
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.token_embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = torch.nn.Parameter(torch.empty(1, self.context_length + num_cls_tokens, embed_dim))
        self.causal_masking = causal_masking
        if self.causal_masking:
            mask = _build_causal_attention_mask(self.context_length)
            # register the mask as a buffer so it can be moved to the right device
            self.register_buffer('mask', mask)

        self.supply_seq_len_to_head = supply_seq_len_to_head
        self.num_cls_tokens = num_cls_tokens
        self.embed_dim = embed_dim
        if num_cls_tokens > 0:
            assert self.causal_masking is False, "Masking + CLS token isn't implemented"
            self.cls_token = torch.nn.Parameter(torch.zeros(1, self.num_cls_tokens, embed_dim))

        self.init_parameters(init_param_style)

    @torch.no_grad()
    def init_parameters(self, init_param_style='openclip'):
        # OpenCLIP style initialization
        torch.nn.init.normal_(self.token_embedding.weight, std=0.02)
        torch.nn.init.normal_(self.pos_embed, std=0.01)

        if init_param_style == 'openclip':
            # OpenCLIP style initialization
            scale = self.embed_dim**-0.5
            if self.num_cls_tokens > 0:
                torch.nn.init.normal_(self.cls_token)
                self.cls_token *= scale
        elif init_param_style == 'vit':
            self.cls_token.data.fill_(0)
        else:
            raise ValueError(f'Unknown init {init_param_style}')

    def forward(self, text: torch.Tensor) -> dict:
        # text tokens are of shape B x L x D
        text_tokens = self.token_embedding(text)
        # concat CLS tokens if any
        if self.num_cls_tokens > 0:
            B = text_tokens.shape[0]
            class_tokens = self.cls_token.expand(B, -1, -1)  # stole class_tokens impl from Phil Wang, thanks
            text_tokens = torch.cat((class_tokens, text_tokens), dim=1)
        text_tokens = text_tokens + self.pos_embed
        return_dict: dict[str, dict] = {
            'trunk': {
                'tokens': text_tokens,
            },
            'head': {},
        }
        # Compute sequence length after adding CLS tokens
        if self.supply_seq_len_to_head:
            text_lengths = text.argmax(dim=-1)
            return_dict['head'] = {
                'seq_len': text_lengths,
            }
        if self.causal_masking:
            return_dict['trunk'].update({'attn_mask': self.mask})
        return return_dict


class Im2Video(torch.nn.Module):
    """Convert an image into a trivial video."""

    def __init__(self, time_dim: int = 2):
        super().__init__()
        self.time_dim = time_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            # B, C, H, W -> B, C, T, H, W
            return x.unsqueeze(self.time_dim)
        if x.ndim == 5:
            return x
        raise ValueError(f'Dimension incorrect {x.shape}')


class PadIm2Video(Im2Video):
    def __init__(self, ntimes: int, pad_type: Literal['zero', 'repeat'], time_dim: int = 2):
        super().__init__(time_dim=time_dim)
        assert ntimes > 0
        self.ntimes = ntimes
        self.pad_type = pad_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        if x.shape[self.time_dim] == 1:
            if self.pad_type == 'repeat':
                new_shape = [1] * len(x.shape)
                new_shape[self.time_dim] = self.ntimes
                x = x.repeat(new_shape)
            elif self.pad_type == 'zero':
                padarg = [0, 0] * len(x.shape)
                padarg[2 * self.time_dim + 1] = self.ntimes - x.shape[self.time_dim]
                x = torch.nn.functional.pad(x, padarg)
        return x
