from abc import ABC
from functools import partial

import torch
from einops import rearrange
from pydantic import BaseModel
from timm.layers import LayerNorm, LayerNorm2d
from timm.models.regnet import RegStage

from turbo_alignment.modeling.multimodal.projectors.registry import (
    MultiModalProjectorRegistry,
)
from turbo_alignment.settings.modality import ModalityProjectorType


class HoneybeeVisualProjectorConfig(BaseModel):
    projector_type: str = 'c-abs'
    initializer_range: float = 1.0
    depth: int = 3
    mlp_depth: int = 2
    hidden_size: int = 1024
    num_eos_tokens: int = 0
    pos_emb: bool = True
    feature_layer_index: int = -1  # vision feature layer index; -1: last layer

    prenorm: bool = False  # if True, LN is applied to vision features

    encoder_hidden_size: int | None = None
    output_hidden_size: int | None = None


def build_pos_embeds(config: HoneybeeVisualProjectorConfig, num_input_tokens: int, vision_hidden_size: int):
    if config.pos_emb:
        pos_emb = torch.nn.Parameter(torch.zeros(1, num_input_tokens, vision_hidden_size))
        torch.nn.init.trunc_normal_(pos_emb, mean=0.0, std=0.02)
    else:
        pos_emb = None

    return pos_emb


def build_eos_tokens(config: HoneybeeVisualProjectorConfig, output_hidden_size: int):
    # think tokens
    num_eos_tokens = config.num_eos_tokens
    if num_eos_tokens:
        eos_tokens = torch.nn.Parameter(torch.randn(1, num_eos_tokens, output_hidden_size))
        torch.nn.init.trunc_normal_(eos_tokens, mean=0.0, std=config.initializer_range)
    else:
        eos_tokens = None

    return eos_tokens


def build_prenorm(config: HoneybeeVisualProjectorConfig):
    if getattr(config, 'prenorm', False):
        prenorm = LayerNorm(config.encoder_hidden_size)
    else:
        prenorm = None
    return prenorm


def build_mlp(depth: int, hidden_size: int, output_hidden_size: int):
    layers: list[torch.nn.Module] = [torch.nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        layers.append(torch.nn.SiLU())
        layers.append(torch.nn.Linear(output_hidden_size, output_hidden_size))
    return torch.nn.Sequential(*layers)


class Projector(torch.nn.Module):
    """Base projector class"""

    def __init__(
        self,
        encoder_hidden_size: int,
        text_hidden_size: int,
        n_modality_embs: int,
        config: HoneybeeVisualProjectorConfig = HoneybeeVisualProjectorConfig(),
    ):
        super().__init__()
        self.config = config

        self.config.output_hidden_size = text_hidden_size
        self.config.encoder_hidden_size = encoder_hidden_size

        # think tokens
        self.eos_tokens = build_eos_tokens(config, self.config.output_hidden_size)
        self.num_input_tokens = n_modality_embs

        # pos emb
        self.pos_emb = build_pos_embeds(config, self.num_input_tokens, self.config.encoder_hidden_size)

        self.prenorm = build_prenorm(config)

        self.build_net()

    def build_net(self):
        raise NotImplementedError()

    def _forward(self, x):
        raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, encoder_hidden_size) tensor from the visual backbone (CLIP visual encoder),
                including cls token.
        """
        if self.prenorm is not None:
            x = self.prenorm(x)

        if self.pos_emb is not None:
            x += self.pos_emb

        x = self._forward(x)  # (B, L, output_hidden_size)

        B = x.size(0)
        if self.eos_tokens is not None:
            x = torch.cat([x, self.eos_tokens.expand(B, -1, -1)], dim=1)

        return x

    def _load_from_state_dict(self, state_dict, *args, **kwargs):
        # update old ckpt compatible with current code
        # pos_emb = state_dict['abstractor.pos_emb']

        pos_emb = state_dict['pos_emb']
        if pos_emb.size(1) == self.pos_emb.size(1) + 1:
            # remove obsolete first pos emb (for cls token originally)
            # state_dict['abstractor.pos_emb'] = pos_emb[:, 1:]
            state_dict['pos_emb'] = pos_emb[:, 1:]

        super()._load_from_state_dict(state_dict, *args, **kwargs)


class ConvProjector(ABC, Projector):
    def _forward(self, x):
        # x: [B, L, dim]
        hw = int(x.size(1) ** 0.5)
        x = rearrange(x, 'b (h w) d -> b d h w', h=hw, w=hw)
        x = self.net(x)
        x = rearrange(x, 'b d h w -> b (h w) d')
        x = self.readout(x)

        return x


@MultiModalProjectorRegistry.register(ModalityProjectorType.C_ABSTRACTOR)
class CAbstractor(ConvProjector):
    """C-Abstractor based on RegBlock"""

    def build_net(self):
        encoder_hidden_size = self.config.encoder_hidden_size
        hidden_size = self.config.hidden_size
        output_hidden_size = self.config.output_hidden_size
        depth = self.config.depth
        mlp_depth = self.config.mlp_depth

        n_queries = self.num_input_tokens
        assert (n_queries**0.5).is_integer(), 'n_queries must be square number'
        hw = int(n_queries**0.5)

        RegBlock = partial(
            RegStage,
            stride=1,
            dilation=1,
            act_layer=torch.nn.SiLU,
            norm_layer=LayerNorm2d,
        )

        s1 = RegBlock(
            depth,
            encoder_hidden_size,
            hidden_size,
        )
        sampler = torch.nn.AdaptiveAvgPool2d((hw, hw))
        s2 = RegBlock(
            depth,
            hidden_size,
            hidden_size,
        )

        if depth:
            self.net = torch.nn.Sequential(s1, sampler, s2)
            self.readout = build_mlp(mlp_depth, hidden_size, output_hidden_size)
        else:
            self.net = sampler
            self.readout = build_mlp(mlp_depth, encoder_hidden_size, output_hidden_size)
