import os

import torch

from .consts import GEMMA_MODEL_PATH, QWEN_MODEL_PATH, QWEN3_MODEL_PATH


def has_two_gpus() -> bool:
    return torch.cuda.is_available() and torch.cuda.device_count() >= 2


def has_model(model_path: str) -> bool:
    return os.path.exists(model_path) and os.path.isdir(model_path)


def has_gemma_model() -> bool:
    return has_model(GEMMA_MODEL_PATH)


def has_qwen_model() -> bool:
    return has_model(QWEN_MODEL_PATH)


def has_qwen3_model() -> bool:
    return has_model(QWEN3_MODEL_PATH)
