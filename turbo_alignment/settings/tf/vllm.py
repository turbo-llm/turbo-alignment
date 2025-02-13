from typing import Literal, Optional

from turbo_alignment.settings.base import ExtraFieldsNotAllowedBaseModel


class EngineSettings(ExtraFieldsNotAllowedBaseModel):
    dtype: Optional[Literal['auto', 'half', 'float16', 'bfloat16', 'float', 'float32']] = 'bfloat16'
    tensor_parallel_size: int = 1
    gpu_memory_utilization: Optional[float] = 0.95
    max_logprobs: Optional[int] = None
    enable_lora: bool = False
    max_lora_rank: Optional[int] = None
