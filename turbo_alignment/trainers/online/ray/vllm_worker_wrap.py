import torch
import torch.distributed
from vllm.worker.worker import Worker


def stateless_init_process_group(master_address, master_port, rank, world_size, device):
    """
    vLLM provides `StatelessProcessGroup` to create a process group
    without considering the global process group in torch.distributed.
    It is recommended to create `StatelessProcessGroup`, and then initialize
    the data-plane communication (NCCL) between external (train processes)
    and vLLM workers.
    """
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    pg = StatelessProcessGroup.create(host=master_address, port=master_port, rank=rank, world_size=world_size)
    pynccl = PyNcclCommunicator(pg, device=device)
    return pynccl


class WorkerWrap(Worker):
    def init_weight_update_group(
        self,
        master_address,
        master_port,
        rank_offset,
        world_size,
    ):
        from vllm.distributed.parallel_state import get_world_group

        rank = get_world_group().rank + rank_offset
        self.model_update_group = stateless_init_process_group(
            master_address,
            master_port,
            rank,
            world_size,
            self.device,
        )

    def update_weight(self, name, dtype, shape):
        weight = torch.empty(shape, dtype=dtype, device='cuda')
        self.model_update_group.broadcast(
            weight,
            src=0,
            # src=torch.cuda.current_device(),
            stream=torch.cuda.current_stream(),
        )

        self.model_runner.model.load_weights(weights=[(name, weight)])

        del weight
