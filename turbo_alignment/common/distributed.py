import os
from typing import Literal

import numpy as np
import torch
import torch.distributed

world_size = int(os.getenv('WORLD_SIZE', '1'))


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
    tensor: torch.Tensor,
    name: str,
    train_eval: Literal['train', 'eval'] | None = None,
    use_global: bool = True,
) -> dict[str, float]:
    if use_global and torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        # On each rank, compute local sums on GPU
        local_sum = tensor.sum()
        local_sq_sum = (tensor * tensor).sum()
        local_count = torch.tensor([tensor.numel()], dtype=torch.long, device=tensor.device)

        # Stack them into one tensor so we only do one all_reduce
        packed = torch.cat([local_sum.unsqueeze(0), local_sq_sum.unsqueeze(0), local_count.float()])
        # e.g. packed = [local_sum, local_sq_sum, local_count]

        torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)
        # Now packed = [global_sum, global_sq_sum, global_count]

        global_sum = packed[0]
        global_sq_sum = packed[1]
        global_count = packed[2]

        # Mean and variance on GPU
        global_mean = global_sum / global_count
        # population variance: E[x^2] - E[x]^2
        var = (global_sq_sum / global_count) - (global_mean * global_mean)
        global_std = var.sqrt()

    else:
        # No distributed or single rank
        global_mean = tensor.mean()
        global_std = tensor.std()

    # NOTE: Converting to Python float forces a sync. If you want to avoid sync
    # completely, return the GPU tensors. But typically for logging you do want floats.
    mean_val = global_mean.detach()  # .cpu().item()
    std_val = global_std.detach()  # .cpu().item()

    # if use_global:
    #     mean = get_global_mean(tensor)
    #     std = get_global_std(tensor, mean=mean)
    # else:
    #     mean = torch.mean(tensor)
    #     std = torch.std(tensor)

    metrics = {}
    if train_eval is not None:
        metrics[f'{train_eval}/{name}_mean'] = mean_val
        metrics[f'{train_eval}/{name}_std'] = std_val
    else:
        metrics[f'{name}_mean'] = mean_val
        metrics[f'{name}_std'] = std_val

    return metrics
