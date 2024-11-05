from enum import Enum

# TODO fix vllm_num_engines,vllm_tensor_parallel_size bacause it is enum
class vLLMActorType(str, Enum):
    DISTRIBUTED_VLLM = 'distributed_vllm'
    
    vllm_num_engines: int = 1
    vllm_tensor_parallel_size: int = 1

class HFActorType(str, Enum):
    LOCAL_TRANSFORMERS = 'local_transformers'


class CriticType(str, Enum):
    LOCAL_TRANSFORMERS = 'local_transformers'
    RAY_TRANSFORMERS = 'ray_transformers'
    DISTRIBUTED_VLLM = 'distributed_vllm'


class RewardProcessorType(str, Enum):
    RLOO = 'rloo'
