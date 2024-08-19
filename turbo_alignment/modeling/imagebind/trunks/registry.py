from abc import ABC
from functools import partial

import torch

from turbo_alignment.modeling.common.helpers import EinOpsRearrange
from turbo_alignment.modeling.common.transformer import (
    MultiheadAttention,
    SimpleTransformer,
)
from turbo_alignment.modeling.imagebind.models import (
    ImageBindArchitectureSettings,
    ModalityType,
)


class Trunks(ABC, torch.nn.ModuleDict):
    @staticmethod
    def __instantiate_trunk(
        embed_dim: int,
        num_blocks: int,
        num_heads: int,
        pre_transformer_ln: bool,
        add_bias_kv: bool,
        drop_path_rate: float,
    ) -> SimpleTransformer:
        return SimpleTransformer(
            embed_dim=embed_dim,
            num_blocks=num_blocks,
            ffn_dropout_rate=0.0,
            drop_path_rate=drop_path_rate,
            attn_target=partial(
                MultiheadAttention,
                embed_dim=embed_dim,
                num_heads=num_heads,
                bias=True,
                add_bias_kv=add_bias_kv,
            ),
            pre_transformer_layer=torch.nn.Sequential(
                torch.nn.LayerNorm(embed_dim, eps=1e-6) if pre_transformer_ln else torch.nn.Identity(),
                EinOpsRearrange('b l d -> l b d'),
            ),
            post_transformer_layer=EinOpsRearrange('l b d -> b l d'),
        )

    @staticmethod
    def __get_vision_trunk(vision_embed_dim: int, vision_num_blocks: int, vision_num_heads: int) -> SimpleTransformer:
        return Trunks.__instantiate_trunk(
            vision_embed_dim,
            vision_num_blocks,
            vision_num_heads,
            pre_transformer_ln=True,
            add_bias_kv=False,
            drop_path_rate=0.0,
        )

    @staticmethod
    def __get_text_trunk(
        text_embed_dim: int,
        text_num_blocks: int,
        text_num_heads: int,
    ) -> SimpleTransformer:
        return Trunks.__instantiate_trunk(
            text_embed_dim,
            text_num_blocks,
            text_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=False,
            drop_path_rate=0.0,
        )

    @staticmethod
    def __get_audio_trunk(
        audio_embed_dim: int,
        audio_num_blocks: int,
        audio_num_heads: int,
        audio_drop_path_rate: float,
    ) -> SimpleTransformer:
        return Trunks.__instantiate_trunk(
            audio_embed_dim,
            audio_num_blocks,
            audio_num_heads,
            pre_transformer_ln=False,
            add_bias_kv=True,
            drop_path_rate=audio_drop_path_rate,
        )

    def __init__(self, settings: ImageBindArchitectureSettings):
        super().__init__()

        self[ModalityType.VISION] = self.__get_vision_trunk(
            settings.vision_embed_dim, settings.vision_num_blocks, settings.vision_num_heads
        )
        self[ModalityType.TEXT] = self.__get_text_trunk(
            settings.text_embed_dim,
            settings.text_num_blocks,
            settings.text_num_heads,
        )
        self[ModalityType.AUDIO] = self.__get_audio_trunk(
            settings.audio_embed_dim,
            settings.audio_num_blocks,
            settings.audio_num_heads,
            settings.audio_drop_path_rate,
        )
