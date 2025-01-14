# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from typing import Any, Tuple
from torch import Tensor
from torch.nn import Module

import deepspeed.comm as dist


class _SeqAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, group: torch.distributed.ProcessGroup, input: Tensor, scatter_idx: int, gather_idx: int
    ) -> Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx

        seq_world_size = dist.get_world_size(group)

        input_list = [t.contiguous() for t in torch.tensor_split(input, seq_world_size, scatter_idx)]
        rank0_print(
            f'BeginOfSeq2Seq {dist.get_rank()=} {input.size()=} {scatter_idx=} {input_list[0].size()=} {input_list[1].size()=}'  # noqa: E501
        )
        output_list = [torch.empty_like(input_list[0]) for _ in range(seq_world_size)]
        # TODO Use all_to_all_single instead
        dist.all_to_all(output_list, input_list, group=group)
        r = torch.cat(output_list, dim=gather_idx).contiguous()
        rank0_print(f'EndOfSeq2Seq {dist.get_rank()=} {r.size()=}')
        return r

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, None, None]:
        return (None, _SeqAllToAll.apply(ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx), None, None)


def print_with_rank(*args, **kwargs):
    print(f'{dist.get_rank()=}', *args, **kwargs)


def rank0_print(*args, **kwargs):
    if dist.get_rank() == 0:
        print(*args, **kwargs)


class DistributedAttention(torch.nn.Module):
    """Initialization.

    Arguments:
        local_attention (Module): local attention with q,k,v
        sequence_process_group (ProcessGroup): sequence parallel process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
    """

    def __init__(
        self,
        local_attention: Module,
        sequence_process_group: dist.ProcessGroup,
        scatter_idx: int = 2,
        gather_idx: int = 0,
    ) -> None:
        super(DistributedAttention, self).__init__()
        self.local_attn = local_attention
        self.spg = sequence_process_group
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx

    def forward(self, query: Tensor, key: Tensor, value: Tensor, *args: Any) -> Tensor:
        """forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args

        Returns:
            * output (Tensor): context output
        """
        # TODO Merge three alltoall calls into one
        # in shape : e.g.,  [s/p:h:]
        print_with_rank(f'{self.spg.name}')
        print_with_rank(f'Attn Before: {dist.get_rank()=} {query.size()=} {key.size()=} {value.size()=}')
        print_with_rank(f'Attn {dist.get_rank()=} send q')
        query_layer = _SeqAllToAll.apply(self.spg, query, self.scatter_idx, self.gather_idx)
        print_with_rank(f'Attn {dist.get_rank()=} send k')
        key_layer = _SeqAllToAll.apply(self.spg, key, self.scatter_idx, self.gather_idx)
        print_with_rank(f'Attn {dist.get_rank()=} send v')
        value_layer = _SeqAllToAll.apply(self.spg, value, self.scatter_idx, self.gather_idx)

        # out shape : e.g., [s:h/p:]
        print_with_rank(
            f'Attn After {dist.get_rank()=} {query_layer.size()=} {key_layer.size()=} {value_layer.size()=}'
        )
        context_layer = self.local_attn(query_layer, key_layer, value_layer, *args)
        print_with_rank(f'Attn {dist.get_rank()=} {context_layer.size()=}')

        output = _SeqAllToAll.apply(self.spg, context_layer, self.gather_idx, self.scatter_idx)
        print_with_rank(f'Attn {dist.get_rank()=} {output.size()=}')
        # out e.g., [s/p::h]
        return output
