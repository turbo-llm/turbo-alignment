import operator

import torch
import triton
import triton.language as tl

import torch.nn as nn

from turbo_alignment.common.tf.liger_kernels.utils import calculate_settings, compare_version, ensure_contiguous



if compare_version("triton", operator.ge, "3.0.0"):
    try:
        # typical import path with dispatch available
        from triton.language.extra.libdevice import tanh
    except ModuleNotFoundError:
        # for working with NGC containers
        from triton.language.extra.cuda.libdevice import tanh
else:
    from triton.language.math import tanh


@triton.jit
def _geglu_tanh_forward_kernel(
    a, b, c, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    program_id = tl.program_id(0)

    # locate start index
    a += program_id * stride
    b += program_id * stride
    c += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    a_row = tl.load(a + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b + col_offsets, mask=mask, other=0)

    # tanh approximation form of GELU is computed with:
    # 0.5 * a * (1 + tanh(sqrt(2 / pi) * (a + 0.044715 * a^3)))
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2 / pi)
    a_cubed = a_row * a_row * a_row
    tanh_arg = sqrt_2_over_pi * (a_row + 0.044715 * a_cubed)
    tanh_result = tanh(tanh_arg)
    geglu_a = 0.5 * a_row * (1 + tanh_result)
    c_row = geglu_a * b_row
    tl.store(c + col_offsets, c_row, mask=mask)


@triton.jit
def _geglu_tanh_backward_kernel(
    dc, a, b, stride, n_cols: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    program_id = tl.program_id(0)

    # locate start index
    dc += program_id * stride
    a += program_id * stride
    b += program_id * stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dc_row = tl.load(dc + col_offsets, mask=mask, other=0)
    a_row = tl.load(a + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b + col_offsets, mask=mask, other=0)

    # recomputation to save memory
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2 / pi)
    a_cubed = a_row * a_row * a_row
    tanh_arg = sqrt_2_over_pi * (a_row + 0.044715 * a_cubed)
    tanh_result = tanh(tanh_arg)
    geglu_a = 0.5 * a_row * (1 + tanh_result)

    db_row = dc_row * geglu_a

    # Gradient w.r.t. a can be computed with:
    # b * (0.5 * (1 + tanh(z)) + 0.5 * a * (1 - tanh(z)^2) * (sqrt(2/pi) * (1 + 3 * 0.044715 * a^2)))
    # where z = sqrt(2/pi) * (a + 0.044715 * a^3)
    term1 = 0.5 * (1 + tanh_result)
    tanh_sq = tanh_result * tanh_result
    term2 = (
        0.5
        * a_row
        * (1 - tanh_sq)
        * (sqrt_2_over_pi * (1 + 3 * 0.044715 * a_row * a_row))
    )
    da_row = dc_row * b_row * (term1 + term2)

    tl.store(a + col_offsets, da_row, mask=mask)
    tl.store(b + col_offsets, db_row, mask=mask)


def geglu_forward(a, b):
    ori_shape = a.shape

    n_cols = ori_shape[-1]
    a = a.view(-1, n_cols)
    b = b.view(-1, n_cols)
    c = torch.empty_like(a)
    n_rows = a.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _geglu_tanh_forward_kernel[(n_rows,)](
        a,
        b,
        c,
        c.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return a, b, c.view(*ori_shape)


def geglu_backward(a, b, dc):
    ori_shape = dc.shape
    n_cols = ori_shape[-1]
    dc = dc.view(-1, n_cols)
    n_rows = dc.shape[0]

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    _geglu_tanh_backward_kernel[(n_rows,)](
        dc,
        a,
        b,
        dc.stride(-2),
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return a.view(*ori_shape), b.view(*ori_shape)


class LigerGELUMulFunction(torch.autograd.Function):
    @staticmethod
    @ensure_contiguous
    def forward(ctx, a, b):
        a, b, c = geglu_forward(a, b)
        ctx.save_for_backward(a, b)
        return c

    @staticmethod
    @ensure_contiguous
    def backward(ctx, dc):
        a, b = ctx.saved_tensors
        a, b = geglu_backward(a, b, dc)
        return a, b

class LigerGEGLUMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        # TODO: support exact GELU
        # Right now Gemma 1, 1.1 and 2 models are all using `gelu_pytorch_tanh`
        # https://github.com/huggingface/transformers/blob/v4.40.1/src/transformers/models/gemma/modeling_gemma.py#L175
        # https://github.com/huggingface/transformers/blob/v4.40.1/src/transformers/activations.py#L46
        # So we can safely assume we use tanh approximation form all the time

    def forward(self, x):

        return self.down_proj(
            LigerGELUMulFunction.apply(self.gate_proj(x), self.up_proj(x))
        )