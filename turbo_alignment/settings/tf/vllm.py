from typing import Literal

from turbo_alignment.settings.base import ExtraFieldsNotAllowedBaseModel


class EngineSettings(ExtraFieldsNotAllowedBaseModel):
    dtype: Literal['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'] | None = 'bfloat16'
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float | None = 0.95
    max_logprobs: int | None = None
    enable_lora: bool = False
    max_lora_rank: int | None = None
