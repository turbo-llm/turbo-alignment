from datetime import timedelta
from typing import Any, Literal

import numpy as np
import torch
import os
import torch.distributed
from torch.distributed.distributed_c10d import (
    Backend,
    PrefixStore,
    Store,
    _new_process_group_helper,
    _world,
    default_pg_timeout,
    rendezvous,
)


world_size = int(os.getenv("WORLD_SIZE", "1"))


def get_global_mean(values: torch.Tensor) -> float:
    # Calculate the mean reward for the current process
    local_sum = values.sum().item()

    if world_size == 1:
        return values.mean().item()

    num_rewards = torch.tensor(len(values), device=values.device)

    # Create a tensor to hold the global mean reward
    global_sum = torch.tensor(local_sum, device=values.device)

    # Collect mean rewards from all processes
    torch.distributed.all_reduce(global_sum, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(num_rewards, op=torch.distributed.ReduceOp.SUM)
    global_mean = (global_sum / num_rewards).item()

    return global_mean


def get_global_std(values: torch.Tensor, mean: float) -> float:
    local_sum = ((values - mean) ** 2).sum().item()

    if world_size == 1:
        return local_sum

    # Calculate the mean reward for the current process
    num_rewards = torch.tensor(len(values), device=values.device)

    # Create a tensor to hold the global mean reward
    global_sum = torch.tensor(local_sum, device=values.device)

    # Collect mean rewards from all processes
    torch.distributed.all_reduce(global_sum, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(num_rewards, op=torch.distributed.ReduceOp.SUM)
    global_mean = np.sqrt((global_sum / (num_rewards - 1)).item())

    return global_mean


def get_log_mean_std(
    tensor: torch.Tensor, name: str, train_eval: Literal['train', 'eval'] | None = None
) -> dict[str, float]:
    mean = get_global_mean(tensor)
    metrics = {}
    if train_eval is not None:
        metrics[f'{train_eval}/{name}_mean'] = mean
        metrics[f'{train_eval}/{name}_std'] = get_global_std(tensor, mean=mean)
    else:
        metrics[f'{name}_mean'] = mean
        metrics[f'{name}_std'] = get_global_std(tensor, mean=mean)

    return metrics


# Copy from pytorch to allow creating multiple main groups.
# https://github.com/pytorch/pytorch/blob/main/torch/distributed/distributed_c10d.py
def init_process_group(
    backend: str | Backend | None = None,
    init_method: str | None = None,
    timeout: timedelta | None = None,
    world_size: int = -1,
    rank: int = -1,
    store: Store | None = None,
    group_name: str | None = None,
    pg_options: Any = None,
):
    assert (store is None) or (init_method is None), 'Cannot specify both init_method and store.'

    if store is not None:
        assert world_size > 0, 'world_size must be positive if using store'
        assert rank >= 0, 'rank must be non-negative if using store'
    elif init_method is None:
        init_method = 'env://'

    if backend:
        backend = Backend(backend)
    else:
        backend = Backend('undefined')

    if timeout is None:
        timeout = default_pg_timeout

    # backward compatible API
    if store is None:
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)

        # Use a PrefixStore to avoid accidental overrides of keys used by
        # different systems (e.g. RPC) in case the store is multi-tenant.
        store = PrefixStore(group_name, store)

    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        pg_options=pg_options,
        timeout=timeout,
    )

    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}

    return pg