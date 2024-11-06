from enum import Enum
from turbo_alignment.settings.base import ExtraFieldsNotAllowedBaseModel
from typing import Literal

class ActorType(str, Enum):
    DISTRIBUTED_VLLM = 'distributed_vllm'
    LOCAL_TRANSFORMERS = 'local_transformers'
    
class vLLMActorSettings(ExtraFieldsNotAllowedBaseModel):
    actor_type: Literal[ActorType.DISTRIBUTED_VLLM] = ActorType.DISTRIBUTED_VLLM
    
    vllm_num_engines: int = 1
    vllm_tensor_parallel_size: int = 1

class HFActorSettings(ExtraFieldsNotAllowedBaseModel):
    actor_type: Literal[ActorType.LOCAL_TRANSFORMERS] = ActorType.LOCAL_TRANSFORMERS

class CriticType(str, Enum):
    LOCAL_TRANSFORMERS = 'local_transformers'
    RAY_TRANSFORMERS = 'ray_transformers'
    DISTRIBUTED_VLLM = 'distributed_vllm'


class RewardProcessorType(str, Enum):
    RLOO = 'rloo'
