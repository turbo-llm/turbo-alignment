from abc import ABC, abstractmethod

import torch


class BaseModalityEncoder(torch.nn.Module, ABC):
    @abstractmethod
    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        ...

    @property
    @abstractmethod
    def emb_dim(self) -> int:
        ...

    @property
    @abstractmethod
    def device(self) -> int:
        ...

    @property
    @abstractmethod
    def n_modality_embs(self) -> int:
        ...
