# pylint: skip-file

import torch
import torch.distributed as dist


class GatherAllLogits(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: torch.Tensor, sp_group: dist.ProcessGroup) -> torch.Tensor:
        """
        Gather all logits accross rank in sequence parallel group
        args:
            logits: tensor with size batch_size x (seq_len // sequence parall size) x vocab_size
            sp_group: sequence parallel group
        return:
            tensor with size batch_size x seq_len x vocab_size
        """

        sp_world_size = dist.get_world_size(sp_group)
        sp_rank = dist.get_rank(sp_group)
        ctx.sp_world_size = sp_world_size
        ctx.sp_rank = sp_rank
        ctx.seqlen = logits.size(1) * sp_world_size

        bs = logits.size(0)
        transposed = logits.transpose(0, 1).contiguous()
        all_logits = torch.zeros(
            (ctx.seqlen, bs) + logits.shape[2:],
            dtype=logits.dtype,
            device=logits.device,
        )

        dist.all_gather_into_tensor(all_logits, transposed, group=sp_group)
        del transposed
        return all_logits.transpose(0, 1)

    @staticmethod
    def backward(ctx, grad_output):
        step_seqlen = ctx.seqlen // ctx.sp_world_size
        sp_rank = ctx.sp_rank
        grad_output_part = grad_output[:, step_seqlen * sp_rank : step_seqlen * (sp_rank + 1), :]
        return grad_output_part, None
