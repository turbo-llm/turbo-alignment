from abc import ABC
from functools import partial

import torch

from turbo_alignment.modeling.imagebind.models import (
    ImageBindArchitectureSettings,
    ModalityType,
)
from turbo_alignment.modeling.imagebind.preprocessors.impl import (
    AudioPreprocessor,
    PadIm2Video,
    PatchEmbedGeneric,
    RGBDTPreprocessor,
    SpatioTemporalPosEmbeddingHelper,
    TextPreprocessor,
)


class Preprocessors(ABC, torch.nn.ModuleDict):
    @staticmethod
    def __get_rgbt_preprocessor(
        video_frames: int,
        kernel_size: tuple[int, int, int],
        stride: tuple[int, int, int],
        vision_embed_dim: int,
    ) -> RGBDTPreprocessor:
        rgbt_stem = PatchEmbedGeneric(
            proj_stem=[
                PadIm2Video(pad_type='repeat', ntimes=2),
                torch.nn.Conv3d(
                    in_channels=3,
                    kernel_size=kernel_size,
                    out_channels=vision_embed_dim,
                    stride=stride,
                    bias=False,
                ),
            ]
        )

        return RGBDTPreprocessor(
            img_size=[3, video_frames, 224, 224],
            num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            rgbt_stem=rgbt_stem,
            depth_stem=None,
        )

    @staticmethod
    def __get_text_preprocessor(
        text_embed_dim: int,
    ) -> TextPreprocessor:
        return TextPreprocessor(
            context_length=77,
            vocab_size=49408,
            embed_dim=text_embed_dim,
            causal_masking=True,
        )

    @staticmethod
    def __get_audio_preprocessor(
        audio_kernel_size: int,
        audio_stride: int,
        audio_embed_dim: int,
        audio_num_mel_bins: int,
        audio_target_len: int,
    ):
        audio_stem = PatchEmbedGeneric(
            proj_stem=[
                torch.nn.Conv2d(
                    in_channels=1,
                    kernel_size=audio_kernel_size,
                    stride=audio_stride,
                    out_channels=audio_embed_dim,
                    bias=False,
                ),
            ],
            norm_layer=torch.nn.LayerNorm(normalized_shape=audio_embed_dim),
        )

        return AudioPreprocessor(
            img_size=[1, audio_num_mel_bins, audio_target_len],
            num_cls_tokens=1,
            pos_embed_fn=partial(SpatioTemporalPosEmbeddingHelper, learnable=True),
            audio_stem=audio_stem,
        )

    def __init__(self, settings: ImageBindArchitectureSettings):
        super().__init__()

        video_frames: int = 2

        self[ModalityType.VISION] = self.__get_rgbt_preprocessor(
            video_frames,
            settings.rgbt_kernel_size,
            settings.rgbt_stride,
            settings.vision_embed_dim,
        )
        self[ModalityType.TEXT] = self.__get_text_preprocessor(settings.text_embed_dim)
        self[ModalityType.AUDIO] = self.__get_audio_preprocessor(
            settings.audio_kernel_size,
            settings.audio_stride,
            settings.audio_embed_dim,
            settings.audio_num_mel_bins,
            settings.audio_target_len,
        )
