from typing import Any

import torch
import torch.distributed as dist

from turbo_alignment.sequence_parallel.collator import pad_for_sequence_parallel


def all_gather_variable(tensor: torch.Tensor, group=None) -> list[torch.Tensor]:
    world_size = dist.get_world_size()
    # Gather shapes first
    shape = torch.as_tensor(tensor.shape, device=tensor.device)
    shapes = [torch.empty_like(shape) for _ in range(world_size)]
    dist.all_gather(shapes, shape, group=group)
    # Gather data
    inputs = [tensor] * world_size
    outputs = [torch.empty(*_shape, dtype=tensor.dtype, device=tensor.device) for _shape in shapes]
    dist.all_to_all(outputs, inputs, group=group)
    return outputs


def gather_and_split(
    tensor: torch.Tensor,
    group: dist.ProcessGroup,
    dim: int = -1,
    pad_value: Any = None,
    padding_side: str = 'right',
):
    r = torch.distributed.get_rank(group)
    group_size = torch.distributed.get_world_size(group)
    all_gathered = all_gather_variable(tensor, group)
    single_tensor = torch.cat(all_gathered, dim=dim)
    single_tensor = pad_for_sequence_parallel(single_tensor, group_size, pad_value, dim=dim, padding_side=padding_side)
    chunk_size = single_tensor.size(dim) // group_size
    return single_tensor.narrow(dim, chunk_size * r, chunk_size)
