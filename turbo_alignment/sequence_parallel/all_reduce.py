from torch.autograd.function import Function

import torch.distributed as dist


class _AllReduce(Function):
    # pylint: disable=abstract-method

    @staticmethod
    def forward(ctx, op, group, tensor):  # pylint: disable=arguments-differ
        ctx.group = group
        ctx.op = op
        tensor = tensor.clone()
        dist.all_reduce(tensor, op=op, group=group)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):  # pylint: disable=arguments-differ
        return (None, None) + (_AllReduce.apply(ctx.op, ctx.group, grad_output),)


def all_reduce(tensor, op, group):
    return _AllReduce.apply(op, group, tensor)
