from abc import ABC, abstractmethod
from pathlib import Path

import torch


class BaseModalityReader(ABC):
    @abstractmethod
    def read(self, path: str) -> torch.Tensor:
        ...
