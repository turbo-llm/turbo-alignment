import contextlib

from turbo_alignment.sequence_parallel.accelerate import (
    AcceleratorWithSequenceParallelism,
    HfTrainerDeepSpeedSeqPConfig,
)


@contextlib.contextmanager
def patch_acclerator():
    import transformers.trainer
    import transformers.integrations.deepspeed

    old_accleator_cls = transformers.trainer.Accelerator
    old_config_cls = transformers.integrations.deepspeed.HfTrainerDeepSpeedConfig
    try:
        transformers.trainer.Accelerator = AcceleratorWithSequenceParallelism
        transformers.integrations.deepspeed.HfTrainerDeepSpeedConfig = HfTrainerDeepSpeedSeqPConfig
        yield

    finally:
        transformers.trainer.Accelerator = old_accleator_cls
        transformers.integrations.deepspeed.HfTrainerDeepSpeedConfig = old_config_cls
