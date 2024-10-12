from enum import Enum


class LLMActorType(str, Enum):
    LOCAL_TRANSFORMERS = 'local_transformers'
    DISTRIBUTED_VLLM = 'distributed_vllm'


class CriticType(str, Enum):
    LOCAL_TRANSFORMERS = 'local_transformers'
    DISTRIBUTED_VLLM = 'distributed_vllm'


class RewardProcessorType(str, Enum):
    RLOO = 'rloo'

