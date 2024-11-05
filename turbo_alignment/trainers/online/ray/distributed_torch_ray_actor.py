import logging
import os
import ray
import socket
import torch
import deepspeed
import random
import numpy as np
from datetime import timedelta
from torch import distributed as dist

class DistributedTorchRayActor:
    def __init__(self, world_size, rank, local_rank, master_addr, master_port):
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            level=logging.INFO,
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self._world_size = world_size
        self._rank = rank
        self._local_rank = local_rank
        self._master_addr = master_addr if master_addr else self._get_current_node_ip()
        self._master_port = master_port if master_port else self._get_free_port()
        os.environ["MASTER_ADDR"] = self._master_addr
        os.environ["MASTER_PORT"] = str(self._master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)
        # NOTE: Ray will automatically set the CUDA_VISIBLE_DEVICES
        # environment variable for each actor, so always set device to 0
        # os.environ["LOCAL_RANK"] = str(self._local_rank)
        os.environ["LOCAL_RANK"] = "0"

    @staticmethod
    def _get_current_node_ip():
        address = ray._private.services.get_node_ip_address()
        # strip ipv6 address
        return address.strip("[]")

    @staticmethod
    def _get_free_port():
        with socket.socket() as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]

    def get_master_addr_port(self):
        return self._master_addr, self._master_port
    
    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    #TODO change seed
    def _setup_distributed(self, timeout=timedelta(minutes=30)):
        self.set_seed(seed=0)
        # if self.args.local_rank == -1 and "LOCAL_RANK" in os.environ:  # for slurm
        #     self.args.local_rank = int(os.environ["LOCAL_RANK"])

        # if self.args.local_rank != -1:
        #     torch.cuda.set_device(self.args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        deepspeed.init_distributed(timeout=timeout)
        self.world_size = dist.get_world_size()
    
    def init_model_from_pretrained(self, *args, **kwargs):
        raise NotImplementedError()