# pylint: skip-file

from typing import Tuple, TYPE_CHECKING

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from typing import Any

    from torch import Tensor


class _SeqAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: 'Any', group: dist.ProcessGroup, input: 'Tensor', scatter_idx: int, gather_idx: int) -> 'Tensor':
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx

        seq_world_size = dist.get_world_size(group)

        input_list = [t.contiguous() for t in torch.tensor_split(input, seq_world_size, scatter_idx)]
        output_list = [torch.empty_like(input_list[0]) for _ in range(seq_world_size)]
        # TODO Use all_to_all_single instead
        dist.all_to_all(output_list, input_list, group=group)
        return torch.cat(output_list, dim=gather_idx).contiguous()

    @staticmethod
    def backward(ctx: 'Any', *grad_output: 'Tensor') -> Tuple[None, 'Tensor', None, None]:
        return (None, _SeqAllToAll.apply(ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx), None, None)
