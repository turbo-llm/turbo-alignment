from abc import ABC

import torch.nn

from turbo_alignment.common.registry import Registrable


class Critic(ABC, Registrable):

    def __init__(self):
        ...


    def generate_responses(self, input_ids: torch.Tensor):
        ...
