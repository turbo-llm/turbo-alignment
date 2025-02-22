from pathlib import Path
from typing import Any

from turbo_alignment.settings.base import ExtraFieldsNotAllowedBaseModel


class TrainerSettings(ExtraFieldsNotAllowedBaseModel):
    evaluation_strategy: str = 'steps'
    save_strategy: str = 'steps'
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 32
    eval_steps: int = 150
    save_steps: int = 150
    logging_steps: int = 5
    learning_rate: float = 0.0003
    num_train_epochs: int = 3
    max_steps: int = -1
    lr_scheduler_type: str = 'cosine'
    lr_scheduler_kwargs: dict[str, Any] = {}
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    fp16: bool = True
    bf16: bool = False
    tf32: bool = False
    torch_compile: bool = False
    optim: str = 'adamw_torch'
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    deepspeed: Path | None = None
    save_total_limit: int = 1
    save_only_model: bool = False
    no_cuda: bool = False
    prediction_loss_only: bool = False
    load_best_model_at_end: bool = True
    logging_first_step: bool = True
    fsdp_config: dict[str, Any] | None = None
    fsdp: str | list[str] | None = ''
    dataloader_num_workers: int = 8
    dataloader_prefetch_factor: int | None = None
    dataloader_persistent_workers: bool | None = False
    dataloader_pin_memory: bool | None = True
    gradient_checkpointing: bool = False
    gradient_checkpointing_kwargs: dict[str, Any] = {}
    neftune_noise_alpha: float | None = None
    report_to: list[str] = []
