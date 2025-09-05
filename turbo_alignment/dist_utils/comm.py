# mypy: ignore-errors

import torch
import torch.distributed as dist


def create_and_broadcast(
    tensor: torch.Tensor | None,
    src: int,
    group: dist.ProcessGroup | None,
    device: 'torch.Device',
):
    """
    All the arguments are the same as of torch.distributed.broadcast
    """
    rank = dist.get_rank()
    object_list = [None, None]
    if rank == src:
        object_list = [tuple(tensor.size()), tensor.dtype]

    dist.broadcast_object_list(
        object_list,
        src=src,
        group=group,
        device=device,
    )

    if rank == src:
        result = tensor

    else:
        size, dtype = object_list
        result = torch.zeros(size, dtype=dtype, device=device)

    dist.broadcast(
        result,
        src=src,
        group=group,
    )

    return result
