import os

import torch

from .consts import MODEL_PATH


def has_two_gpus() -> bool:
    return torch.cuda.is_available() and torch.cuda.device_count() >= 2


def has_gemma_model() -> bool:
    return os.path.exists(MODEL_PATH) and os.path.isdir(MODEL_PATH)
