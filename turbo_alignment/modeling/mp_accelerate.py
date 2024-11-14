import math

import torch
from accelerate.accelerator import (
    AcceleratedScheduler,
    Accelerator,
    LRScheduler,
    is_deepspeed_available,
)
from accelerate.logging import get_logger
from turbo_alignment.modeling import parallel_states as mpu

# based on https://github.com/huggingface/accelerate/pull/2877/files

if is_deepspeed_available():
    from accelerate.utils import (
        DeepSpeedEngineWrapper,
        DeepSpeedOptimizerWrapper,
        DeepSpeedSchedulerWrapper,
        DummyOptim,
        DummyScheduler,
    )

logger = get_logger(__name__)


class AcceleratorWithModelParallism(Accelerator):
    def _prepare_deepspeed(self, *args):
        import deepspeed

        deepspeed_plugin = self.state.deepspeed_plugin

        is_dataloader_present = any(isinstance(obj, torch.utils.data.DataLoader) for obj in args)
        result = [
            self._prepare_one(obj, first_pass=True) if isinstance(obj, torch.utils.data.DataLoader) else obj
            for obj in args
        ]

        ### BEGIN OF PATCH
        if mpu.sequence_parallel_is_initialized():
            world_size = mpu.get_data_parallel_world_size()
        else:
            world_size = self.num_processes
        ### ENF OF PATCH

        if deepspeed_plugin.is_auto("train_micro_batch_size_per_gpu"):
            if is_dataloader_present:
                batch_sizes = [obj.batch_size for obj in args if hasattr(obj, "batch_size")]
                if any(bs is None for bs in batch_sizes):
                    raise ValueError(
                        "At least one of the dataloaders passed to `accelerate.prepare()` has `None` as batch size. "
                        "Please set an integer value in `train_micro_batch_size_per_gpu` in the deepspeed config file "
                        "or assign integer value to `AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu']`."
                    )
                if self.split_batches:
                    batch_sizes = [batch_size // world_size for batch_size in batch_sizes]  ## PATCH

                batch_size_per_device = min(batch_sizes) if deepspeed_plugin.is_train_batch_min else max(batch_sizes)
                if len(batch_sizes) > 1:
                    logger.info(
                        "Since you passed both train and evaluation dataloader, `is_train_batch_min` (here "
                        f"{deepspeed_plugin.is_train_batch_min} will decide the `train_batch_size` ({batch_size_per_device})."
                    )
            else:
                raise ValueError(
                    "When using DeepSpeed, `accelerate.prepare()` requires you to pass at least one of training or evaluation dataloaders "
                    "with `batch_size` attribute returning an integer value "
                    "or alternatively set an integer value in `train_micro_batch_size_per_gpu` in the deepspeed config file "
                    "or assign integer value to `AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu']`."
                )
        else:
            batch_size_per_device = deepspeed_plugin.get_value("train_micro_batch_size_per_gpu")

        # handle `gradient_accumulation_steps` when the value is `auto`
        deepspeed_plugin.fill_match(
            "gradient_accumulation_steps",
            must_match=False,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
        )

        config_kwargs = {
            "train_micro_batch_size_per_gpu": batch_size_per_device,
            "train_batch_size": batch_size_per_device
            * deepspeed_plugin.get_value("gradient_accumulation_steps")
            * world_size,  ### PATCHED
            "gradient_clipping": 1.0,
            "zero_optimization.stage3_gather_16bit_weights_on_model_save": False,
        }

        model = None
        optimizer = None
        scheduler = None
        for obj in result:
            if isinstance(obj, torch.nn.Module):
                model = obj
            elif isinstance(obj, (torch.optim.Optimizer, DummyOptim)):
                optimizer = obj
            elif (isinstance(obj, (LRScheduler, DummyScheduler))) or (
                type(obj).__name__ in deepspeed.runtime.lr_schedules.VALID_LR_SCHEDULES
            ):
                scheduler = obj

        if optimizer is not None:
            if "optimizer" in deepspeed_plugin.deepspeed_config and not isinstance(optimizer, (DummyOptim)):
                raise ValueError(
                    "You cannot specify an optimizer in the config file and in the code at the same time. "
                    "Please remove the optimizer from the config file or "
                    "create `accelerate.utils.DummyOptim` in the code."
                )
            elif "optimizer" not in deepspeed_plugin.deepspeed_config and isinstance(optimizer, (DummyOptim)):
                raise ValueError(
                    "You cannot create a `DummyOptim` without specifying an optimizer in the config file."
                )

            if isinstance(optimizer, (torch.optim.Optimizer)):
                deepspeed_plugin.deepspeed_config["zero_allow_untested_optimizer"] = True

        if scheduler is not None:
            if "scheduler" in deepspeed_plugin.deepspeed_config and not isinstance(scheduler, (DummyScheduler)):
                raise ValueError(
                    "You cannot specify a scheduler in the config file and in the code at the same time. "
                    "Please remove the scheduler from the config file or "
                    "create `accelerate.utils.DummyScheduler` in the code."
                )
            elif (
                "scheduler" not in deepspeed_plugin.deepspeed_config
                and isinstance(scheduler, (DummyScheduler))
                and scheduler.lr_scheduler_callable is None
            ):
                raise ValueError(
                    "Either specify a scheduler in the config file or "
                    "pass in the `lr_scheduler_callable` parameter when using `accelerate.utils.DummyScheduler`."
                )

        if optimizer is not None and scheduler is not None:
            if isinstance(optimizer, (DummyOptim)) and not isinstance(scheduler, (DummyScheduler)):
                raise ValueError(
                    "You can only specify `accelerate.utils.DummyScheduler` in the code when using "
                    "`accelerate.utils.DummyOptim`."
                )

        if model is not None:
            # deal with config keys that use `auto` value and rely on model's hidden_size
            hidden_size_based_keys = [
                "zero_optimization.reduce_bucket_size",
                "zero_optimization.stage3_prefetch_bucket_size",
                "zero_optimization.stage3_param_persistence_threshold",
            ]
            hidden_size_auto_keys = [x for x in hidden_size_based_keys if deepspeed_plugin.is_auto(x)]
            if len(hidden_size_auto_keys) > 0:
                reasoning = (
                    "therefore it's not possible to automatically fill out the following `auto` entries "
                    + f"in the DeepSpeed config file: {hidden_size_auto_keys}. You can fix that by replacing "
                    + "`auto` values for these keys with an integer value of your choice."
                )
                if not hasattr(model, "config"):
                    raise ValueError("Can't find `model.config` entry, " + reasoning)

                if hasattr(model.config, "hidden_size"):
                    hidden_size = model.config.hidden_size
                elif hasattr(model.config, "hidden_sizes"):
                    # if there are many hidden sizes pick the largest one
                    hidden_size = max(model.config.hidden_sizes)
                else:
                    raise ValueError(
                        "Can find neither `model.config.hidden_size` nor `model.config.hidden_sizes`, " + reasoning
                    )

                config_kwargs.update(
                    {
                        "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                        "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                    }
                )

            if isinstance(optimizer, (DummyOptim)):
                config_kwargs.update(
                    {"optimizer.params.lr": optimizer.lr, "optimizer.params.weight_decay": optimizer.weight_decay}
                )
            if isinstance(scheduler, (DummyScheduler)) and scheduler.lr_scheduler_callable is None:
                max_lr = (
                    getattr(scheduler.optimizer, "lr", None)
                    if getattr(scheduler.optimizer, "defaults", None) is None
                    else scheduler.optimizer.defaults["lr"]
                )
                config_kwargs.update(
                    {
                        "scheduler.params.warmup_min_lr": 0,
                        "scheduler.params.warmup_max_lr": max_lr,
                        "scheduler.params.warmup_num_steps": scheduler.warmup_num_steps,
                    }
                )
                if scheduler.total_num_steps is not None:
                    config_kwargs["scheduler.params.total_num_steps"] = (
                        math.ceil(scheduler.total_num_steps / world_size)
                        if not self.split_batches
                        else scheduler.total_num_steps
                    )

            deepspeed_plugin.deepspeed_config_process(must_match=False, mismatches=[], **config_kwargs)
            self.deepspeed_config = deepspeed_plugin.deepspeed_config
            kwargs = dict(model=model, config_params=self.deepspeed_config)

            #### BEGIN OF PATCH
            if mpu.sequence_parallel_is_initialized():
                logger.info('Set mpu')
                kwargs["mpu"] = mpu
            else:
                logger.info('Does not set mpu')
            #### END OF PATCH

            if optimizer is not None:
                if isinstance(optimizer, (DummyOptim)):
                    kwargs["model_parameters"] = optimizer.params
                    if isinstance(scheduler, (DummyScheduler)) and scheduler.lr_scheduler_callable is not None:
                        kwargs["lr_scheduler"] = scheduler.lr_scheduler_callable
                else:
                    if self.deepspeed_config["zero_optimization"].get("offload_optimizer", {}).get(
                        "device", "none"
                    ) != "none" and self.deepspeed_config.get("zero_force_ds_cpu_optimizer", True):
                        from deepspeed.ops.adam import DeepSpeedCPUAdam

                        defaults = {k: v for k, v in optimizer.defaults.items() if k in ["lr", "weight_decay"]}
                        optimizer = DeepSpeedCPUAdam(optimizer.param_groups, **defaults)
                    kwargs["optimizer"] = optimizer
                    if scheduler is not None:
                        if type(scheduler).__name__ in deepspeed.runtime.lr_schedules.VALID_LR_SCHEDULES:
                            kwargs["lr_scheduler"] = scheduler

            engine, optimizer, _, lr_scheduler = deepspeed.initialize(**kwargs)
            if optimizer is not None:
                optimizer = DeepSpeedOptimizerWrapper(optimizer)
            if scheduler is not None:
                if lr_scheduler is None:
                    scheduler = AcceleratedScheduler(
                        scheduler,
                        optimizer,
                        step_with_optimizer=self.step_scheduler_with_optimizer,
                        split_batches=self.split_batches,
                    )
                else:
                    scheduler = DeepSpeedSchedulerWrapper(lr_scheduler, optimizer)

            for i in range(len(result)):
                if isinstance(result[i], torch.nn.Module):
                    result[i] = engine
                elif isinstance(result[i], (torch.optim.Optimizer, DummyOptim)):
                    result[i] = optimizer
                elif (isinstance(result[i], (LRScheduler, DummyScheduler))) or (
                    type(result[i]).__name__ in deepspeed.runtime.lr_schedules.VALID_LR_SCHEDULES
                ):
                    result[i] = scheduler
            # pointing for deepspeed_engine_wrapped.backward()
            self.deepspeed_engine_wrapped = DeepSpeedEngineWrapper(engine)
            self._models.append(engine)
            if optimizer is not None:
                self._optimizers.append(optimizer)
            if scheduler is not None:
                self._schedulers.append(scheduler)
            if len(self._models) > 1:
                raise AssertionError(
                    "You can't use same `Accelerator()` instance with multiple models when using DeepSpeed"
                )
        return tuple(result)
