from abc import ABC

import torch

from turbo_alignment.modeling.common.helpers import SelectElement, SelectEOSAndProject
from turbo_alignment.modeling.imagebind.models import (
    ImageBindArchitectureSettings,
    ModalityType,
)


class Heads(ABC, torch.nn.ModuleDict):
    @staticmethod
    def __get_vision_head(
        vision_embed_dim: int,
        out_embed_dim: int,
    ) -> torch.nn.Sequential:
        return torch.nn.Sequential(
            torch.nn.LayerNorm(normalized_shape=vision_embed_dim, eps=1e-6),
            SelectElement(index=0),
            torch.nn.Linear(vision_embed_dim, out_embed_dim, bias=False),
        )

    @staticmethod
    def __get_text_head(
        text_embed_dim: int,
        out_embed_dim: int,
    ) -> SelectEOSAndProject:
        return SelectEOSAndProject(
            proj=torch.nn.Sequential(
                torch.nn.LayerNorm(normalized_shape=text_embed_dim, eps=1e-6),
                torch.nn.Linear(text_embed_dim, out_embed_dim, bias=False),
            )
        )

    @staticmethod
    def __get_audio_head(
        audio_embed_dim: int,
        out_embed_dim: int,
    ) -> torch.nn.Sequential:
        return torch.nn.Sequential(
            torch.nn.LayerNorm(normalized_shape=audio_embed_dim, eps=1e-6),
            SelectElement(index=0),
            torch.nn.Linear(audio_embed_dim, out_embed_dim, bias=False),
        )

    def __init__(self, settings: ImageBindArchitectureSettings):
        super().__init__()

        self[ModalityType.VISION] = self.__get_vision_head(settings.vision_embed_dim, settings.out_embed_dim)
        self[ModalityType.TEXT] = self.__get_text_head(settings.text_embed_dim, settings.out_embed_dim)
        self[ModalityType.AUDIO] = self.__get_audio_head(settings.audio_embed_dim, settings.out_embed_dim)
