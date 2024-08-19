from abc import ABC

import torch

from turbo_alignment.modeling.common.helpers import LearnableLogitScaling, Normalize
from turbo_alignment.modeling.imagebind.models import (
    ImageBindArchitectureSettings,
    ModalityType,
)


class Postprocessors(ABC, torch.nn.ModuleDict):
    def __init__(self, _settings: ImageBindArchitectureSettings):
        super().__init__()

        self[ModalityType.VISION] = Normalize(dim=-1)

        self[ModalityType.TEXT] = torch.nn.Sequential(Normalize(dim=-1), LearnableLogitScaling(learnable=True))

        self[ModalityType.AUDIO] = torch.nn.Sequential(
            Normalize(dim=-1),
            LearnableLogitScaling(logit_scale_init=20.0, learnable=False),
        )
