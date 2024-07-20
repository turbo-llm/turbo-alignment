from abc import ABC
from pathlib import Path

import torch
from torch.nn import Module
from transformers import PreTrainedModel

from turbo_alignment.modeling.multimodal.encoders.base import BaseModalityEncoder
from turbo_alignment.settings.modality import Modality, ModalityProjectorType


class BaseMultiModalModeling(Module, ABC):
    def __init__(
        self,
        language_model: PreTrainedModel,
        encoders: dict[Modality, BaseModalityEncoder],
        n_modality_embs: int,
        modality_projector_mapping: dict[Modality, ModalityProjectorType | None],
        modality_projector_initialization_mapping: dict[Modality, Path | None] | None,
        peft: bool = True,  # TODO: Pre-training without PEFT adapters
    ) -> None:
        """
        :param n_modality_embs: The number of projections. If 1, the whole\
            image will be projected into one embedding of lm
        :param peft: If True, assume that lm was initialized with peft
        """
        super().__init__()

        self.language_model = language_model
        self.n_modality_embs = n_modality_embs
        self.modality_projector_mapping = modality_projector_mapping
        self.modality_projector_initialization_mapping = modality_projector_initialization_mapping
        self.peft = peft

        self.language_model_dim = (
            language_model.base_model.model.model.embed_tokens.modules_to_save.default.weight.shape[1]
        )
        self.encoders = torch.nn.ModuleDict(encoders)  # type: ignore[arg-type]

        for encoder in self.encoders.values():
            encoder.eval()

            for param in encoder.parameters():
                param.requires_grad = False
