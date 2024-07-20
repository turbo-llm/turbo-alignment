from abc import ABC

import einops
import numpy as np
import torch


class Normalize(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(x, dim=self.dim, p=2)


class LearnableLogitScaling(torch.nn.Module):
    def __init__(
        self,
        logit_scale_init: float = 1 / 0.07,
        learnable: bool = True,
        max_logit_scale: float = 100,
    ) -> None:
        super().__init__()
        self.max_logit_scale = max_logit_scale
        self.logit_scale_init = logit_scale_init
        self.learnable = learnable
        log_logit_scale = torch.ones([]) * np.log(self.logit_scale_init)
        if learnable:
            self.log_logit_scale = torch.nn.Parameter(log_logit_scale)
        else:
            self.register_buffer('log_logit_scale', log_logit_scale)

    def forward(self, x: torch.Tensor):
        return torch.clip(self.log_logit_scale.exp(), max=self.max_logit_scale) * x

    def extra_repr(self):
        st = (
            f'logit_scale_init={self.logit_scale_init},learnable={self.learnable},'
            f' max_logit_scale={self.max_logit_scale}'
        )
        return st


class EinOpsRearrange(torch.nn.Module):
    def __init__(self, rearrange_expr: str) -> None:
        super().__init__()
        self.rearrange_expr = rearrange_expr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einops.rearrange(x, self.rearrange_expr)


class VerboseNNModule(torch.nn.Module, ABC):
    """
    Wrapper around nn.Module that prints registered buffers and parameter names.
    """

    @staticmethod
    def get_readable_tensor_repr(name: str, t: torch.Tensor) -> str:
        return f'({name}): tensor({t.shape}, requires_grad={t.requires_grad})\n'

    def extra_repr(self) -> str:
        named_modules = list(set(p[0] for p in self.named_modules()))

        string_repr: str = ''
        for name, value in self.named_parameters():
            if name.split('.')[0] not in named_modules:
                string_repr += self.get_readable_tensor_repr(name, value)

        for name, value in self.named_buffers():
            string_repr += self.get_readable_tensor_repr(name.split('.')[0], value)

        return string_repr


def cast_if_src_dtype(
    tensor: torch.Tensor, src_dtype: torch.dtype, tgt_dtype: torch.dtype
) -> tuple[torch.Tensor, bool]:
    updated = False
    if tensor.dtype == src_dtype:
        tensor = tensor.to(dtype=tgt_dtype)
        updated = True
    return tensor, updated


class SelectElement(torch.nn.Module):
    def __init__(self, index: int) -> None:
        super().__init__()
        self.index = index

    def forward(self, x: torch.Tensor):
        assert x.ndim >= 3
        return x[:, self.index, ...]


class SelectEOSAndProject(torch.nn.Module):
    """
    Text Pooling used in OpenCLIP
    """

    def __init__(self, proj: torch.nn.Module) -> None:
        super().__init__()
        self.proj = proj

    def forward(self, x, seq_len):
        assert x.ndim == 3
        # x is of shape B x L x D
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), seq_len]
        x = self.proj(x)
        return x
