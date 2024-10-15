# mypy: ignore-errors
# pylint: skip-file
import torch
import triton
import triton.language as tl
from torch.nn import CrossEntropyLoss


@triton.jit
def liger_cross_entropy_kernel(
    X_ptr,
    X_stride,
    Y_ptr,
    Y_stride,
    loss_ptr,
    loss_stride,
    n_cols,
    n_non_ignore,
    ignore_index,
    label_smoothing: tl.constexpr,
    reduction: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)

    Y_ptr += program_id * Y_stride
    y = tl.load(Y_ptr)

    X_ptr += program_id * X_stride

    if y == ignore_index:
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(X_ptr + X_offsets, 0.0, mask=X_offsets < n_cols)
        return

    loss_ptr += program_id * loss_stride

    m = float('-inf')
    d = 0.0
    ori_X_y = tl.load(X_ptr + y)

    scaled_x_sum = 0.0
    eps = label_smoothing / n_cols

    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols, other=float('-inf'))
        block_max = tl.max(X_block)
        if label_smoothing > 0:
            scaled_x_sum += tl.sum(tl.where(X_offsets < n_cols, -eps * X_block, 0.0))
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(X_block - m_new))
        m = m_new

    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols, other=float('-inf'))
        if reduction == 'mean':
            X_block = (tl.exp(X_block - m) / d - eps) / (n_non_ignore)
        else:
            X_block = tl.exp(X_block - m) / d - eps

        tl.store(X_ptr + X_offsets, X_block, mask=X_offsets < n_cols)

    tl.debug_barrier()

    loss = -(ori_X_y - m - tl.log(d))

    if label_smoothing > 0:
        smooth_loss = scaled_x_sum + label_smoothing * (m + tl.log(d))
        loss = loss * (1 - label_smoothing) + smooth_loss

    if reduction == 'mean':
        loss = loss / n_non_ignore

    X_y = tl.load(X_ptr + y)
    if reduction == 'mean':
        X_y += -(1 - label_smoothing) / (n_non_ignore)
    else:
        X_y += -(1 - label_smoothing)

    tl.store(loss_ptr, loss)
    tl.store(X_ptr + y, X_y)


MAX_FUSED_SIZE = 65536 // 2


@triton.jit
def element_mul_kernel(
    X_ptr,
    X_stride,
    grad_output_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)

    X_ptr += program_id * X_stride

    grad_output = tl.load(grad_output_ptr)

    for i in range(0, n_cols, BLOCK_SIZE):
        X_offsets = i + tl.arange(0, BLOCK_SIZE)
        X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols)
        tl.store(X_ptr + X_offsets, X_block * grad_output, mask=X_offsets < n_cols)


def cross_entropy_forward(_input, target, ignore_index, label_smoothing, reduction):
    BT, V = _input.shape
    n_rows = BT

    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

    loss_1d = torch.zeros(n_rows, dtype=_input.dtype, device=_input.device)

    n_non_ignore = (target != ignore_index).sum().item()

    if _input.stride(-1) != 1:
        _input = _input.contiguous()
    if target.stride(-1) != 1:
        target = target.contiguous()

    liger_cross_entropy_kernel[(n_rows,)](
        X_ptr=_input,
        X_stride=_input.stride(-2),
        Y_ptr=target,
        Y_stride=target.stride(-1),
        loss_ptr=loss_1d,
        loss_stride=loss_1d.stride(-1),
        n_cols=V,
        n_non_ignore=n_non_ignore,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        reduction=reduction,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=32,
    )

    loss = torch.sum(loss_1d)
    return loss, _input


def cross_entropy_backward(_input, grad_output):
    if torch.equal(grad_output, torch.tensor(1.0, device=grad_output.device)):
        pass

    else:
        BT, V = _input.shape
        n_rows = BT
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(V))

        element_mul_kernel[(n_rows,)](
            _input,
            _input.stride(-2),
            grad_output,
            V,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=32,
        )

    return _input


class LigerCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, _input, target, ignore_index=-100, label_smoothing=0.0, reduction='mean'):
        loss, _input = cross_entropy_forward(_input, target, ignore_index, label_smoothing, reduction)
        ctx.save_for_backward(_input.detach())
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        (_input,) = ctx.saved_tensors
        _input = cross_entropy_backward(_input, grad_output)
        return (
            _input,
            None,
            None,
            None,
            None,
        )


class LigerCrossEntropyLoss(CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        super(LigerCrossEntropyLoss, self).__init__(*args, **kwargs)
        assert (self.label_smoothing >= 0) and (
            self.label_smoothing <= 1
        ), f'label_smoothing must be between 0.0 and 1.0. Got: {self.label_smoothing}'
        assert self.reduction in {
            'mean',
            'sum',
            'none',
        }, f"reduction must be one of 'mean', 'sum', or 'none'. Got: {self.reduction}"

    def forward(self, _input, target):
        return LigerCrossEntropyFunction.apply(_input, target, self.ignore_index, self.label_smoothing, self.reduction)
