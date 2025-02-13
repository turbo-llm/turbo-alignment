from typing import Literal, Optional

from turbo_alignment.settings.base import ExtraFieldsNotAllowedBaseModel


class EngineSettings(ExtraFieldsNotAllowedBaseModel):
    dtype: Optional[Literal['auto', 'half', 'float16', 'bfloat16', 'float', 'float32']] = 'auto'
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float | None = 0.9
    max_logprobs: int | None = None
    enable_lora: bool = False
    max_lora_rank: Optional[int] = None
