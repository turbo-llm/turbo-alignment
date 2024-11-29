import contextlib

from turbo_alignment.modeling.mp_accelerate import AcceleratorWithModelParallism, HfTrainerDeepSpeedSeqPConfig
# from turbo_alignment.modeling.mp_pretrained import PretrainedModelWithMPU


@contextlib.contextmanager
def patch_acclerator():
    import transformers.trainer
    import transformers.integrations.deepspeed

    old_accleator_cls = transformers.trainer.Accelerator
    old_config_cls = transformers.integrations.deepspeed.HfTrainerDeepSpeedConfig
    try:
        transformers.trainer.Accelerator = AcceleratorWithModelParallism
        transformers.integrations.deepspeed.HfTrainerDeepSpeedConfig = HfTrainerDeepSpeedSeqPConfig
        yield

    finally:
        transformers.trainer.Accelerator = old_accleator_cls
        transformers.integrations.deepspeed.HfTrainerDeepSpeedConfig = old_config_cls


# @contextlib.contextmanager
# def patch_gemma_base_model():
#     import transformers.models.gemma2.modeling_gemma2

#     old_cls = transformers.models.gemma2.modeling_gemma2.PreTrainedModel

#     try:
#         transformers.models.gemma2.modeling_gemma2.PreTrainedModel = PretrainedModelWithMPU
#         yield
#     finally:
        # transformers.models.gemma2.modeling_gemma2.PreTrainedModel = old_cls


