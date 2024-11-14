import os
from collections import OrderedDict

import torch
from accelerate.accelerator import (
    FSDP_PYTORCH_VERSION,
    Accelerator,
    AcceleratorState,
    AutocastKwargs,
    DataLoaderConfiguration,
    DeepSpeedPlugin,
    DistributedDataParallelKwargs,
    DistributedType,
    DynamoBackend,
    FP8RecipeKwargs,
    FullyShardedDataParallelPlugin,
    GeneralTracker,
    GradientAccumulationPlugin,
    GradientState,
    GradScalerKwargs,
    InitProcessGroupKwargs,
    KwargsHandler,
    LoggerType,
    MegatronLMPlugin,
    PrecisionType,
    ProjectConfiguration,
    RNGType,
    TorchDynamoPlugin,
    _dispatch_batches,
    _even_batches,
    _split_batches,
    _use_seedable_sampler,
    compare_versions,
    filter_trackers,
    is_bf16_available,
    is_deepspeed_available,
    is_megatron_lm_available,
    is_mlu_available,
    is_msamp_available,
    is_npu_available,
    is_torch_version,
    is_torch_xla_available,
    parse_choice_from_env,
    warnings,
    xamp,
    check_os_kernel,
)
from turbo_alignment.modeling import parallel_states

# based on https://github.com/huggingface/accelerate/pull/2877/files

class AcceleratorWithModelParallism(Accelerator):
    def __init__(
        self,
        device_placement: bool = True,
        split_batches: bool = _split_batches,
        mixed_precision: PrecisionType | str | None = None,
        gradient_accumulation_steps: int = 1,
        cpu: bool = False,
        dataloader_config: DataLoaderConfiguration | None = None,
        deepspeed_plugin: DeepSpeedPlugin | None = None,
        fsdp_plugin: FullyShardedDataParallelPlugin | None = None,
        megatron_lm_plugin: MegatronLMPlugin | None = None,
        rng_types: list[str | RNGType] | None = None,
        log_with: str | LoggerType | GeneralTracker | list[str | LoggerType | GeneralTracker] | None = None,
        project_dir: str | os.PathLike | None = None,
        project_config: ProjectConfiguration | None = None,
        gradient_accumulation_plugin: GradientAccumulationPlugin | None = None,
        dispatch_batches: bool | None = _dispatch_batches,
        even_batches: bool = _even_batches,
        use_seedable_sampler: bool = _use_seedable_sampler,
        step_scheduler_with_optimizer: bool = True,
        kwargs_handlers: list[KwargsHandler] | None = None,
        dynamo_backend: DynamoBackend | str | None = None,
    ):
        self.trackers = []
        if project_config is not None:
            self.project_configuration = project_config
        else:
            self.project_configuration = ProjectConfiguration(project_dir=project_dir)
        if project_dir is not None and self.project_dir is None:
            self.project_configuration.set_directories(project_dir)
        if mixed_precision is not None:
            mixed_precision = str(mixed_precision)
            if mixed_precision not in PrecisionType:
                raise ValueError(
                    f"Unknown mixed_precision mode: {mixed_precision}. Choose between {PrecisionType.list()}"
                )

        dynamo_plugin = TorchDynamoPlugin() if dynamo_backend is None else TorchDynamoPlugin(backend=dynamo_backend)

        if deepspeed_plugin is None:  # init from env variables
            deepspeed_plugin = (
                DeepSpeedPlugin() if os.environ.get("ACCELERATE_USE_DEEPSPEED", "false") == "true" else None
            )
        else:
            assert isinstance(
                deepspeed_plugin, DeepSpeedPlugin
            ), "`deepspeed_plugin` must be an `accelerate.utils.DeepSpeedPlugin` object."
            os.environ["ACCELERATE_USE_DEEPSPEED"] = "true"  # use DeepSpeed if plugin is provided
        if deepspeed_plugin:
            if not is_deepspeed_available():
                raise ImportError("DeepSpeed is not installed => run `pip install deepspeed` or build it from source.")
            if is_mlu_available():
                if compare_versions("deepspeed-mlu", "<", "0.10.1"):
                    raise ImportError("DeepSpeed MLU version must be >= 0.10.1. Please update DeepSpeed MLU.")
            elif compare_versions("deepspeed", "<", "0.9.3"):
                raise ImportError("DeepSpeed version must be >= 0.9.3. Please update DeepSpeed.")

            mixed_precision = (
                os.environ.get("ACCELERATE_MIXED_PRECISION", "no") if mixed_precision is None else mixed_precision
            )
            deepspeed_plugin.set_mixed_precision(mixed_precision)
            deepspeed_plugin.set_deepspeed_weakref()

        if os.environ.get("ACCELERATE_USE_FSDP", "false") == "true" or isinstance(
            fsdp_plugin, FullyShardedDataParallelPlugin
        ):
            if is_torch_version("<", FSDP_PYTORCH_VERSION):
                raise ValueError(f"FSDP requires PyTorch >= {FSDP_PYTORCH_VERSION}")

        if fsdp_plugin is None:  # init from env variables
            fsdp_plugin = (
                FullyShardedDataParallelPlugin() if os.environ.get("ACCELERATE_USE_FSDP", "false") == "true" else None
            )
        else:
            if not isinstance(fsdp_plugin, FullyShardedDataParallelPlugin):
                raise TypeError("`fsdp_plugin` must be a FullyShardedDataParallelPlugin object.")
            os.environ["ACCELERATE_USE_FSDP"] = "true"  # use FSDP if plugin is provided

        if megatron_lm_plugin is None:  # init from env variables
            megatron_lm_plugin = (
                MegatronLMPlugin() if os.environ.get("ACCELERATE_USE_MEGATRON_LM", "false") == "true" else None
            )
        else:
            if not isinstance(megatron_lm_plugin, MegatronLMPlugin):
                raise TypeError("`megatron_lm_plugin` must be a MegatronLMPlugin object.")
            os.environ["ACCELERATE_USE_MEGATRON_LM"] = "true"  # use MegatronLM if plugin is provided

        if megatron_lm_plugin:
            if not is_megatron_lm_available():
                raise ImportError("Megatron is not installed. please build it from source.")

        # Kwargs handlers
        self.ddp_handler = None
        self.scaler_handler = None
        self.init_handler = None
        self.fp8_recipe_handler = None
        self.autocast_handler = None
        if kwargs_handlers is not None:
            for handler in kwargs_handlers:
                assert isinstance(
                    handler, KwargsHandler
                ), f"Unsupported kwargs handler passed: {handler}, must be one that inherits `accelerate.utils.KwargsHandler`."
                if isinstance(handler, DistributedDataParallelKwargs):
                    if self.ddp_handler is not None:
                        raise ValueError("You can only pass one `DistributedDataParallelKwargs` in `kwargs_handler`.")
                    else:
                        self.ddp_handler = handler
                elif isinstance(handler, GradScalerKwargs):
                    if self.scaler_handler is not None:
                        raise ValueError("You can only pass one `GradScalerKwargs` in `kwargs_handler`.")
                    else:
                        self.scaler_handler = handler
                elif isinstance(handler, InitProcessGroupKwargs):
                    if self.init_handler is not None:
                        raise ValueError("You can only pass one `InitProcessGroupKwargs` in `kwargs_handler`.")
                    else:
                        self.init_handler = handler
                elif isinstance(handler, FP8RecipeKwargs):
                    if self.fp8_recipe_handler is not None:
                        raise ValueError("You can only pass one `FP8RecipeKwargs` in `kwargs_handler`.")
                    else:
                        self.fp8_recipe_handler = handler
                elif isinstance(handler, AutocastKwargs):
                    if self.autocast_handler is not None:
                        raise ValueError("You can only pass one `AutocastKwargs` in `kwargs_handler`.")
                    else:
                        self.autocast_handler = handler

        kwargs = self.init_handler.to_kwargs() if self.init_handler is not None else {}
        self.state = AcceleratorState(
            mixed_precision=mixed_precision,
            cpu=cpu,
            dynamo_plugin=dynamo_plugin,
            deepspeed_plugin=deepspeed_plugin,
            fsdp_plugin=fsdp_plugin,
            megatron_lm_plugin=megatron_lm_plugin,
            _from_accelerator=True,
            **kwargs,
        )

        if self.fp8_recipe_handler is None and self.state.mixed_precision == "fp8":
            self.fp8_recipe_handler = FP8RecipeKwargs(backend="MSAMP" if is_msamp_available() else "TE")

        trackers = filter_trackers(log_with, self.logging_dir)
        if len(trackers) < 1 and log_with is not None:
            warnings.warn(f"`log_with={log_with}` was passed but no supported trackers are currently installed.")
        self.log_with = trackers

        if (
            (mixed_precision != "bf16")
            and getattr(self.state, "downcast_bfloat", False)
            and (self.state.distributedType != DistributedType.XLA)
        ):
            raise ValueError("Can only use `downcast_bf16` when using `mixed_precision='bf16'` and on a TPU")

        if gradient_accumulation_plugin is not None:
            if gradient_accumulation_steps != 1:
                raise ValueError(
                    "You can only pass one of `gradient_accumulation_steps` and `gradient_accumulation_plugin`. Please only pass in the created `GradientAccumulationPlugin` object."
                )
        else:
            gradient_accumulation_steps = int(
                parse_choice_from_env("ACCELERATE_GRADIENT_ACCUMULATION_STEPS", gradient_accumulation_steps)
            )
            gradient_accumulation_plugin = GradientAccumulationPlugin(num_steps=gradient_accumulation_steps)
        self.gradient_state = GradientState(
            gradient_accumulation_plugin=gradient_accumulation_plugin,
        )

        self.device_placement = device_placement
        if dataloader_config is None:
            dataloader_config = DataLoaderConfiguration()
        self.dataloader_config = dataloader_config
        # Deal with deprecated args
        # TODO: Remove in v1.0.0
        deprecated_dl_args = {}
        if dispatch_batches is not _dispatch_batches:
            deprecated_dl_args["dispatch_batches"] = dispatch_batches
            self.dataloader_config.dispatch_batches = dispatch_batches
        if split_batches is not _split_batches:
            deprecated_dl_args["split_batches"] = split_batches
            self.dataloader_config.split_batches = split_batches
        if even_batches is not _even_batches:
            deprecated_dl_args["even_batches"] = even_batches
            self.dataloader_config.even_batches = even_batches
        if use_seedable_sampler is not _use_seedable_sampler:
            deprecated_dl_args["use_seedable_sampler"] = use_seedable_sampler
            self.dataloader_config.use_seedable_sampler = use_seedable_sampler
        if len(deprecated_dl_args) > 0:
            values = ", ".join([f"{k}={v}" for k, v in deprecated_dl_args.items()])
            warnings.warn(
                f"Passing the following arguments to `Accelerator` is deprecated and will be removed in version 1.0 of Accelerate: {deprecated_dl_args.keys()}. "
                "Please pass an `accelerate.DataLoaderConfiguration` instead: \n"
                f"dataloader_config = DataLoaderConfiguration({values})",
                FutureWarning,
            )
        self.step_scheduler_with_optimizer = step_scheduler_with_optimizer

        # Mixed precision attributes
        self.scaler = None
        self.native_amp = False
        if (
            self.state.mixed_precision == "fp16"
            and self.device.type != "cpu"
            and self.distributed_type not in (DistributedType.DEEPSPEED, DistributedType.MEGATRON_LM)
        ):
            self.native_amp = True
            if self.device.type not in ("xpu", "cuda", "mps", "npu", "xla", "mlu") or is_torch_xla_available(
                check_is_tpu=True
            ):
                raise ValueError(f"fp16 mixed precision requires a GPU (not {self.device.type!r}).")
            kwargs = self.scaler_handler.to_kwargs() if self.scaler_handler is not None else {}
            if self.distributed_type == DistributedType.FSDP:
                from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

                self.scaler = ShardedGradScaler(**kwargs)
            elif is_torch_xla_available(check_is_gpu=True):
                self.scaler = xamp.GradScaler(**kwargs)
            elif is_mlu_available():
                self.scaler = torch.mlu.amp.GradScaler(**kwargs)
            elif is_npu_available():
                self.scaler = torch.npu.amp.GradScaler(**kwargs)
            else:
                self.scaler = torch.cuda.amp.GradScaler(**kwargs)

        elif self.state.mixed_precision == "bf16" and self.distributed_type not in (
            DistributedType.DEEPSPEED,
            DistributedType.MEGATRON_LM,
        ):
            if self.device.type in ["cpu", "xpu"]:
                self.native_amp = True
            else:
                self.native_amp = is_bf16_available(True)
            if mixed_precision == "bf16" and not self.native_amp and not is_torch_xla_available():
                raise ValueError("bf16 mixed precision requires PyTorch >= 1.10 and a supported device.")

        # Start of internal step tracking
        self.step = 0

        # Internal references to the training objects
        self._optimizers = []
        self._models = []
        self._schedulers = []
        self._dataloaders = []
        self._custom_objects = []

        # Hooks
        self._load_model_state_pre_hook = OrderedDict()
        self._save_model_state_pre_hook = OrderedDict()

        # RNG Types
        self.rng_types = rng_types
        if self.rng_types is None:
            self.rng_types = ["generator"]

        # Set a flag tensor for early stopping and other breakpoints
        self.flag_tensor = None

        check_os_kernel()
