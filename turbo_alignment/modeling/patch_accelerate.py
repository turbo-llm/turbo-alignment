import contextlib

from turbo_alignment.modeling.mp_accelerate import AcceleratorWithModelParallism
# from turbo_alignment.modeling.mp_pretrained import PretrainedModelWithMPU


@contextlib.contextmanager
def patch_acclerator():
    import transformers.trainer

    old_accleator_cls = transformers.trainer.Accelerator
    try:
        transformers.trainer.Accelerator = AcceleratorWithModelParallism
        yield

    finally:
        transformers.trainer.Accelerator = old_accleator_cls


# @contextlib.contextmanager
# def patch_gemma_base_model():
#     import transformers.models.gemma2.modeling_gemma2

#     old_cls = transformers.models.gemma2.modeling_gemma2.PreTrainedModel

#     try:
#         transformers.models.gemma2.modeling_gemma2.PreTrainedModel = PretrainedModelWithMPU
#         yield
#     finally:
        # transformers.models.gemma2.modeling_gemma2.PreTrainedModel = old_cls


