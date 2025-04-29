# pylint: disable=line-too-long,superfluous-parens,abstract-method,raise-missing-from,no-else-raise,protected-access,unnecessary-pass,unbalanced-tuple-unpacking,unexpected-keyword-arg,attribute-defined-outside-init
# flake8: noqa

import copy
from typing import Type

import torch
from transformers.modeling_utils import (
    ACCELERATE_MIN_VERSION,
    set_quantized_state,
    ContextManagers,
    PreTrainedModel,
    SpecificPreTrainedModelType,
    _is_quantized,
    _is_ds_init_called,
    set_zero3_state,
    deepspeed_config,
    init_empty_weights,
    is_accelerate_available,
    is_deepspeed_zero3_enabled,
    logger,
    no_init_weights,
)


class PreTrainedModelWithMPU(PreTrainedModel):
    @classmethod
    def get_init_context(
        cls: Type[SpecificPreTrainedModelType],
        _fast_init=True,
        is_quantized=None,
        _is_ds_init_called=None,
        low_cpu_mem_usage=True,
    ):
        init_contexts = [no_init_weights(_enable=_fast_init)]

        if is_deepspeed_zero3_enabled() and not is_quantized and not _is_ds_init_called:
            import deepspeed
            import turbo_alignment.modeling.parallel_states as mpu

            logger.info("Detected DeepSpeed ZeRO-3: activating zero.init() for this model")
            mpu_inited = mpu.sequence_parallel_is_initialized()
            logger.info('MPU inited')
            init_contexts = [
                deepspeed.zero.Init(config_dict_or_path=deepspeed_config(), mpu=mpu if mpu_inited else None),
                set_zero3_state(),
            ] + init_contexts
        elif low_cpu_mem_usage:
            if not is_accelerate_available():
                raise ImportError(
                    f"Using `low_cpu_mem_usage=True` or a `device_map` requires Accelerate: `pip install 'accelerate>={ACCELERATE_MIN_VERSION}'`"
                )
            init_contexts.append(init_empty_weights())

        if is_deepspeed_zero3_enabled() and is_quantized:
            init_contexts.append(set_quantized_state())
        return init_contexts

    @classmethod
    def _from_config(cls, config, **kwargs):
        """
        All context managers that the model should be initialized under go here.

        Args:
            torch_dtype (`torch.dtype`, *optional*):
                Override the default `torch.dtype` and load the model under this dtype.
        """
        # when we init a model from within another model (e.g. VLMs) and dispatch on FA2
        # a warning is raised that dtype should be fp16. Since we never pass dtype from within
        # modeling code, we can try to infer it here same way as done in `from_pretrained`
        torch_dtype = kwargs.pop("torch_dtype", torch.get_default_dtype())
        use_flash_attention_2 = kwargs.pop("use_flash_attention_2", False)

        # override default dtype if needed
        dtype_orig = None
        if torch_dtype is not None:
            dtype_orig = cls._set_default_torch_dtype(torch_dtype)

        config = copy.deepcopy(config)  # We do not want to modify the config inplace in _from_config.

        if config._attn_implementation_internal is not None:
            # In this case, the config has been created with the attn_implementation set by the user, which we
            # should respect.
            attn_implementation = config._attn_implementation_internal
        else:
            attn_implementation = None

        config._attn_implementation = kwargs.pop("attn_implementation", attn_implementation)
        if not getattr(config, "_attn_implementation_autoset", False):
            config = cls._autoset_attn_implementation(
                config,
                use_flash_attention_2=use_flash_attention_2,
                check_device_map=False,
                torch_dtype=torch_dtype,
            )

        if is_deepspeed_zero3_enabled() and not _is_quantized and not _is_ds_init_called:
            import deepspeed
            import turbo_alignment.modeling.parallel_states as mpu

            logger.info("Detected DeepSpeed ZeRO-3: activating zero.init() for this model")
            # this immediately partitions the model across all gpus, to avoid the overhead in time
            # and memory copying it on CPU or each GPU first
            init_contexts = [deepspeed.zero.Init(config_dict_or_path=deepspeed_config(), mpu=mpu), set_zero3_state()]
            with ContextManagers(init_contexts):
                model = cls(config, **kwargs)

        else:
            model = cls(config, **kwargs)

        # restore default dtype if it was modified
        if dtype_orig is not None:
            torch.set_default_dtype(dtype_orig)

        return model
