# pylint: disable=line-too-long,superfluous-parens,abstract-method,raise-missing-from,no-else-raise,protected-access,unnecessary-pass,unbalanced-tuple-unpacking,unexpected-keyword-arg,attribute-defined-outside-init
# flake8: noqa

import copy
import inspect
import json
import os
import warnings
from typing import Optional, Type, Union

import torch
from transformers.modeling_utils import (
    ACCELERATE_MIN_VERSION,
    set_quantized_state,
    CONFIG_NAME,
    FLAX_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    WEIGHTS_NAME,
    BitsAndBytesConfig,
    ContextManagers,
    GenerationConfig,
    PretrainedConfig,
    PreTrainedModel,
    SpecificPreTrainedModelType,
    QuantizationMethod,
    _is_quantized,
    _is_ds_init_called,
    set_zero3_state,
    _add_variant,
    auto_conversion,
    cached_file,
    check_tied_parameters_on_same_device,
    deepspeed_config,
    dispatch_model,
    download_url,
    extract_commit_hash,
    find_adapter_config_file,
    find_tied_parameters,
    get_balanced_memory,
    get_checkpoint_shard_files,
    get_max_memory,
    get_state_dict_dtype,
    infer_auto_device_map,
    init_empty_weights,
    is_accelerate_available,
    is_deepspeed_zero3_enabled,
    is_fsdp_enabled,
    is_peft_available,
    is_remote_url,
    is_safetensors_available,
    load_state_dict,
    logger,
    no_init_weights,
    safe_open,
    has_file,
    AutoHfQuantizer,
    is_offline_mode,
    Thread,
    WEIGHTS_INDEX_NAME,
    restore_default_torch_dtype,
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


    # copied and slighlty modified
    # fmt: off
    # @classmethod
    # @restore_default_torch_dtype
    # def from_pretrained(
    #     cls: Type[SpecificPreTrainedModelType],
    #     pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
    #     *model_args,
    #     config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
    #     cache_dir: Optional[Union[str, os.PathLike]] = None,
    #     ignore_mismatched_sizes: bool = False,
    #     force_download: bool = False,
    #     local_files_only: bool = False,
    #     token: Optional[Union[str, bool]] = None,
    #     revision: str = "main",
    #     use_safetensors: Optional[bool] = None,
    #     weights_only: bool = True,
    #     **kwargs,
    # ) -> SpecificPreTrainedModelType:
    #     r"""
    #     Instantiate a pretrained pytorch model from a pre-trained model configuration.

    #     The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
    #     the model, you should first set it back in training mode with `model.train()`.

    #     The warning *Weights from XXX not initialized from pretrained model* means that the weights of XXX do not come
    #     pretrained with the rest of the model. It is up to you to train those weights with a downstream fine-tuning
    #     task.

    #     The warning *Weights from XXX not used in YYY* means that the layer XXX is not used by YYY, therefore those
    #     weights are discarded.

    #     If model weights are the same precision as the base model (and is a supported model), weights will be lazily loaded
    #     in using the `meta` device and brought into memory once an input is passed through that layer regardless of
    #     `low_cpu_mem_usage`.

    #     Parameters:
    #         pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
    #             Can be either:

    #                 - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
    #                 - A path to a *directory* containing model weights saved using
    #                   [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
    #                 - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
    #                   this case, `from_tf` should be set to `True` and a configuration object should be provided as
    #                   `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
    #                   PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
    #                 - A path or url to a model folder containing a *flax checkpoint file* in *.msgpack* format (e.g,
    #                   `./flax_model/` containing `flax_model.msgpack`). In this case, `from_flax` should be set to
    #                   `True`.
    #                 - `None` if you are both providing the configuration and state dictionary (resp. with keyword
    #                   arguments `config` and `state_dict`).
    #         model_args (sequence of positional arguments, *optional*):
    #             All remaining positional arguments will be passed to the underlying model's `__init__` method.
    #         config (`Union[PretrainedConfig, str, os.PathLike]`, *optional*):
    #             Can be either:

    #                 - an instance of a class derived from [`PretrainedConfig`],
    #                 - a string or path valid as input to [`~PretrainedConfig.from_pretrained`].

    #             Configuration for the model to use instead of an automatically loaded configuration. Configuration can
    #             be automatically loaded when:

    #                 - The model is a model provided by the library (loaded with the *model id* string of a pretrained
    #                   model).
    #                 - The model was saved using [`~PreTrainedModel.save_pretrained`] and is reloaded by supplying the
    #                   save directory.
    #                 - The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
    #                   configuration JSON file named *config.json* is found in the directory.
    #         state_dict (`Dict[str, torch.Tensor]`, *optional*):
    #             A state dictionary to use instead of a state dictionary loaded from saved weights file.

    #             This option can be used if you want to create a model from a pretrained configuration but load your own
    #             weights. In this case though, you should check if using [`~PreTrainedModel.save_pretrained`] and
    #             [`~PreTrainedModel.from_pretrained`] is not a simpler option.
    #         cache_dir (`Union[str, os.PathLike]`, *optional*):
    #             Path to a directory in which a downloaded pretrained model configuration should be cached if the
    #             standard cache should not be used.
    #         from_tf (`bool`, *optional*, defaults to `False`):
    #             Load the model weights from a TensorFlow checkpoint save file (see docstring of
    #             `pretrained_model_name_or_path` argument).
    #         from_flax (`bool`, *optional*, defaults to `False`):
    #             Load the model weights from a Flax checkpoint save file (see docstring of
    #             `pretrained_model_name_or_path` argument).
    #         ignore_mismatched_sizes (`bool`, *optional*, defaults to `False`):
    #             Whether or not to raise an error if some of the weights from the checkpoint do not have the same size
    #             as the weights of the model (if for instance, you are instantiating a model with 10 labels from a
    #             checkpoint with 3 labels).
    #         force_download (`bool`, *optional*, defaults to `False`):
    #             Whether or not to force the (re-)download of the model weights and configuration files, overriding the
    #             cached versions if they exist.
    #         resume_download:
    #             Deprecated and ignored. All downloads are now resumed by default when possible.
    #             Will be removed in v5 of Transformers.
    #         proxies (`Dict[str, str]`, *optional*):
    #             A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
    #             'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
    #         output_loading_info(`bool`, *optional*, defaults to `False`):
    #             Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
    #         local_files_only(`bool`, *optional*, defaults to `False`):
    #             Whether or not to only look at local files (i.e., do not try to download the model).
    #         token (`str` or `bool`, *optional*):
    #             The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
    #             the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).
    #         revision (`str`, *optional*, defaults to `"main"`):
    #             The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
    #             git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
    #             identifier allowed by git.

    #             <Tip>

    #             To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>"`.

    #             </Tip>
    #         _fast_init(`bool`, *optional*, defaults to `True`):
    #             Whether or not to disable fast initialization.

    #             <Tip warning={true}>

    #             One should only disable *_fast_init* to ensure backwards compatibility with `transformers.__version__ <
    #             4.6.0` for seeded model initialization. This argument will be removed at the next major version. See
    #             [pull request 11471](https://github.com/huggingface/transformers/pull/11471) for more information.

    #             </Tip>
    #         attn_implementation (`str`, *optional*):
    #             The attention implementation to use in the model (if relevant). Can be any of `"eager"` (manual implementation of the attention), `"sdpa"` (using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)), or `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)). By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"` implementation.

    #         > Parameters for big model inference

    #         low_cpu_mem_usage(`bool`, *optional*):
    #             Tries not to use more than 1x model size in CPU memory (including peak memory) while loading the model.
    #             Generally should be combined with a `device_map` (such as `"auto"`) for best results.
    #             This is an experimental feature and a subject to change at any moment.
    #             </Tip>
    #                 If the model weights are in the same precision as the model loaded in, `low_cpu_mem_usage` (without
    #                 `device_map`) is redundant and will not provide any benefit in regards to CPU memory usage. However,
    #                 this should still be enabled if you are passing in a `device_map`.
    #             </Tip>
    #         torch_dtype (`str` or `torch.dtype`, *optional*):
    #             Override the default `torch.dtype` and load the model under a specific `dtype`. The different options
    #             are:

    #             1. `torch.float16` or `torch.bfloat16` or `torch.float`: load in a specified
    #               `dtype`, ignoring the model's `config.torch_dtype` if one exists. If not specified
    #               - the model will get loaded in `torch.float` (fp32).

    #             2. `"auto"` - A `torch_dtype` entry in the `config.json` file of the model will be
    #               attempted to be used. If this entry isn't found then next check the `dtype` of the first weight in
    #               the checkpoint that's of a floating point type and use that as `dtype`. This will load the model
    #               using the `dtype` it was saved in at the end of the training. It can't be used as an indicator of how
    #               the model was trained. Since it could be trained in one of half precision dtypes, but saved in fp32.

    #             3. A string that is a valid `torch.dtype`. E.g. "float32" loads the model in `torch.float32`, "float16" loads in `torch.float16` etc.

    #             <Tip>

    #             For some models the `dtype` they were trained in is unknown - you may try to check the model's paper or
    #             reach out to the authors and ask them to add this information to the model's card and to insert the
    #             `torch_dtype` entry in `config.json` on the hub.

    #             </Tip>

    #         device_map (`str` or `Dict[str, Union[int, str, torch.device]]` or `int` or `torch.device`, *optional*):
    #             A map that specifies where each submodule should go. It doesn't need to be refined to each
    #             parameter/buffer name, once a given module name is inside, every submodule of it will be sent to the
    #             same device. If we only pass the device (*e.g.*, `"cpu"`, `"cuda:1"`, `"mps"`, or a GPU ordinal rank
    #             like `1`) on which the model will be allocated, the device map will map the entire model to this
    #             device. Passing `device_map = 0` means put the whole model on GPU 0.

    #             To have Accelerate compute the most optimized `device_map` automatically, set `device_map="auto"`. For
    #             more information about each option see [designing a device
    #             map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
    #         max_memory (`Dict`, *optional*):
    #             A dictionary device identifier to maximum memory if using `device_map`. Will default to the maximum memory available for each
    #             GPU and the available CPU RAM if unset.
    #         tp_plan (`str`, *optional*):
    #             A torch tensor parallel plan, see [here](https://pytorch.org/tutorials/intermediate/TP_tutorial.html). Currently, it only accepts
    #             `tp_plan="auto"` to use predefined plan based on the model. Note that if you use it, you should launch your script accordingly with
    #             `torchrun [args] script.py`. This will be much faster than using a `device_map`, but has limitations.
    #         offload_folder (`str` or `os.PathLike`, *optional*):
    #             If the `device_map` contains any value `"disk"`, the folder where we will offload weights.
    #         offload_state_dict (`bool`, *optional*):
    #             If `True`, will temporarily offload the CPU state dict to the hard drive to avoid getting out of CPU
    #             RAM if the weight of the CPU state dict + the biggest shard of the checkpoint does not fit. Defaults to
    #             `True` when there is some disk offload.
    #         offload_buffers (`bool`, *optional*):
    #             Whether or not to offload the buffers with the model parameters.
    #         quantization_config (`Union[QuantizationConfigMixin,Dict]`, *optional*):
    #             A dictionary of configuration parameters or a QuantizationConfigMixin object for quantization (e.g
    #             bitsandbytes, gptq). There may be other quantization-related kwargs, including `load_in_4bit` and
    #             `load_in_8bit`, which are parsed by QuantizationConfigParser. Supported only for bitsandbytes
    #             quantizations and not preferred. consider inserting all such arguments into quantization_config
    #             instead.
    #         subfolder (`str`, *optional*, defaults to `""`):
    #             In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
    #             specify the folder name here.
    #         variant (`str`, *optional*):
    #             If specified load weights from `variant` filename, *e.g.* pytorch_model.<variant>.bin. `variant` is
    #             ignored when using `from_tf` or `from_flax`.
    #         use_safetensors (`bool`, *optional*, defaults to `None`):
    #             Whether or not to use `safetensors` checkpoints. Defaults to `None`. If not specified and `safetensors`
    #             is not installed, it will be set to `False`.
    #         weights_only (`bool`, *optional*, defaults to `True`):
    #             Indicates whether unpickler should be restricted to loading only tensors, primitive types,
    #             dictionaries and any types added via torch.serialization.add_safe_globals().
    #             When set to False, we can load wrapper tensor subclass weights.
    #         key_mapping (`Dict[str, str], *optional*):
    #             A potential mapping of the weight names if using a model on the Hub which is compatible to a Transformers
    #             architecture, but was not converted accordingly.
    #         kwargs (remaining dictionary of keyword arguments, *optional*):
    #             Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
    #             `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
    #             automatically loaded:

    #                 - If a configuration is provided with `config`, `**kwargs` will be directly passed to the
    #                   underlying model's `__init__` method (we assume all relevant updates to the configuration have
    #                   already been done)
    #                 - If a configuration is not provided, `kwargs` will be first passed to the configuration class
    #                   initialization function ([`~PretrainedConfig.from_pretrained`]). Each key of `kwargs` that
    #                   corresponds to a configuration attribute will be used to override said attribute with the
    #                   supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
    #                   will be passed to the underlying model's `__init__` function.

    #     <Tip>

    #     Activate the special ["offline-mode"](https://huggingface.co/transformers/installation.html#offline-mode) to
    #     use this method in a firewalled environment.

    #     </Tip>

    #     Examples:

    #     ```python
    #     >>> from transformers import BertConfig, BertModel

    #     >>> # Download model and configuration from huggingface.co and cache.
    #     >>> model = BertModel.from_pretrained("google-bert/bert-base-uncased")
    #     >>> # Model was saved using *save_pretrained('./test/saved_model/')* (for example purposes, not runnable).
    #     >>> model = BertModel.from_pretrained("./test/saved_model/")
    #     >>> # Update configuration during loading.
    #     >>> model = BertModel.from_pretrained("google-bert/bert-base-uncased", output_attentions=True)
    #     >>> assert model.config.output_attentions == True
    #     >>> # Loading from a TF checkpoint file instead of a PyTorch model (slower, for example purposes, not runnable).
    #     >>> config = BertConfig.from_json_file("./tf_model/my_tf_model_config.json")
    #     >>> model = BertModel.from_pretrained("./tf_model/my_tf_checkpoint.ckpt.index", from_tf=True, config=config)
    #     >>> # Loading from a Flax checkpoint file instead of a PyTorch model (slower)
    #     >>> model = BertModel.from_pretrained("google-bert/bert-base-uncased", from_flax=True)
    #     ```

    #     * `low_cpu_mem_usage` algorithm:

    #     This is an experimental function that loads the model using ~1x model size CPU memory

    #     Here is how it works:

    #     1. save which state_dict keys we have
    #     2. drop state_dict before the model is created, since the latter takes 1x model size CPU memory
    #     3. after the model has been instantiated switch to the meta device all params/buffers that
    #     are going to be replaced from the loaded state_dict
    #     4. load state_dict 2nd time
    #     5. replace the params/buffers from the state_dict

    #     Currently, it can't handle deepspeed ZeRO stage 3 and ignores loading errors

    #     """
    #     state_dict = kwargs.pop("state_dict", None)
    #     from_tf = kwargs.pop("from_tf", False)
    #     from_flax = kwargs.pop("from_flax", False)
    #     _ = kwargs.pop("resume_download", None)
    #     proxies = kwargs.pop("proxies", None)
    #     output_loading_info = kwargs.pop("output_loading_info", False)
    #     use_auth_token = kwargs.pop("use_auth_token", None)
    #     _ = kwargs.pop("trust_remote_code", None)
    #     _ = kwargs.pop("mirror", None)
    #     from_pipeline = kwargs.pop("_from_pipeline", None)
    #     from_auto_class = kwargs.pop("_from_auto", False)
    #     _fast_init = kwargs.pop("_fast_init", True)
    #     torch_dtype = kwargs.pop("torch_dtype", None)
    #     low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", None)
    #     device_map = kwargs.pop("device_map", None)
    #     max_memory = kwargs.pop("max_memory", None)
    #     offload_folder = kwargs.pop("offload_folder", None)
    #     offload_state_dict = kwargs.pop("offload_state_dict", False)
    #     offload_buffers = kwargs.pop("offload_buffers", False)
    #     load_in_8bit = kwargs.pop("load_in_8bit", False)
    #     load_in_4bit = kwargs.pop("load_in_4bit", False)
    #     quantization_config = kwargs.pop("quantization_config", None)
    #     subfolder = kwargs.pop("subfolder", "")
    #     commit_hash = kwargs.pop("_commit_hash", None)
    #     variant = kwargs.pop("variant", None)
    #     adapter_kwargs = kwargs.pop("adapter_kwargs", {})
    #     adapter_name = kwargs.pop("adapter_name", "default")
    #     use_flash_attention_2 = kwargs.pop("use_flash_attention_2", False)
    #     generation_config = kwargs.pop("generation_config", None)
    #     gguf_file = kwargs.pop("gguf_file", None)
    #     tp_plan = kwargs.pop("tp_plan", None)
    #     key_mapping = kwargs.pop("key_mapping", None)

    #     if state_dict is not None and (pretrained_model_name_or_path is not None or gguf_file is not None):
    #         raise ValueError(
    #             "`state_dict` cannot be passed together with a model name or a `gguf_file`. Use one of the two loading strategies."
    #         )

    #     if tp_plan is not None and tp_plan != "auto":
    #         # TODO: we can relax this check when we support taking tp_plan from a json file, for example.
    #         raise ValueError(f"tp_plan supports 'auto' only for now but got {tp_plan}.")
    #     if tp_plan is not None and device_map is not None:
    #         raise ValueError(
    #             "`tp_plan` and `device_map` are mutually exclusive. Choose either one for parallelization."
    #         )

    #     # If torchrun was used, make sure to TP by default. This way people don't need to change tp or device map
    #     if device_map == "auto" and tp_plan is None and int(os.environ.get("WORLD_SIZE", 0)):
    #         tp_plan = "auto"  # device_map = "auto" in torchrun equivalent to TP plan = AUTO!
    #         device_map = None

    #     # We need to correctly dispatch the model on the current process device. The easiest way for this is to use a simple
    #     # `device_map` pointing to the correct device
    #     device_mesh = None
    #     if tp_plan is not None:
    #         if not is_torch_greater_or_equal("2.5"):
    #             raise EnvironmentError("tensor parallel is only supported for `torch>=2.5`.")
    #         if not torch.distributed.is_initialized():
    #             try:
    #                 logger.warning("Tensor Parallel requires torch.distributed to be initialized first.")
    #                 rank = int(os.environ["RANK"])
    #                 world_size = int(os.environ["WORLD_SIZE"])
    #                 torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    #                 torch.cuda.set_device(rank)
    #             except Exception as e:
    #                 raise EnvironmentError(
    #                     "We tried to initialize torch.distributed for you, but it failed, make"
    #                     "sure you init torch distributed in your script to use `tp_plan='auto'`"
    #                 ) from e

    #         # Detect the accelerator on the machine. If no accelerator is available, it returns CPU.
    #         device_type = torch._C._get_accelerator().type
    #         device_module = torch.get_device_module(device_type)
    #         # Get device with index assuming equal number of devices per host
    #         tp_device = torch.device(device_type, torch.distributed.get_rank() % device_module.device_count())
    #         # This is the easiest way to dispatch to the current process device
    #         device_map = tp_device

    #         # Assuming sharding the model onto the world
    #         world_size = torch.distributed.get_world_size()
    #         device_mesh = torch.distributed.init_device_mesh(tp_device.type, (world_size,))

    #     if is_fsdp_enabled():
    #         low_cpu_mem_usage = True

    #     if use_auth_token is not None:
    #         warnings.warn(
    #             "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
    #             FutureWarning,
    #         )
    #         if token is not None:
    #             raise ValueError(
    #                 "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
    #             )
    #         token = use_auth_token

    #     if token is not None and adapter_kwargs is not None and "token" not in adapter_kwargs:
    #         adapter_kwargs["token"] = token

    #     if use_safetensors is None and not is_safetensors_available():
    #         use_safetensors = False

    #     if gguf_file is not None and not is_accelerate_available():
    #         raise ValueError("accelerate is required when loading a GGUF file `pip install accelerate`.")

    #     if commit_hash is None:
    #         if not isinstance(config, PretrainedConfig):
    #             # We make a call to the config file first (which may be absent) to get the commit hash as soon as possible
    #             resolved_config_file = cached_file(
    #                 pretrained_model_name_or_path,
    #                 CONFIG_NAME,
    #                 cache_dir=cache_dir,
    #                 force_download=force_download,
    #                 proxies=proxies,
    #                 local_files_only=local_files_only,
    #                 token=token,
    #                 revision=revision,
    #                 subfolder=subfolder,
    #                 _raise_exceptions_for_gated_repo=False,
    #                 _raise_exceptions_for_missing_entries=False,
    #                 _raise_exceptions_for_connection_errors=False,
    #             )
    #             commit_hash = extract_commit_hash(resolved_config_file, commit_hash)
    #         else:
    #             commit_hash = getattr(config, "_commit_hash", None)

    #     if is_peft_available():
    #         _adapter_model_path = adapter_kwargs.pop("_adapter_model_path", None)

    #         if _adapter_model_path is None:
    #             _adapter_model_path = find_adapter_config_file(
    #                 pretrained_model_name_or_path,
    #                 cache_dir=cache_dir,
    #                 force_download=force_download,
    #                 proxies=proxies,
    #                 local_files_only=local_files_only,
    #                 _commit_hash=commit_hash,
    #                 **adapter_kwargs,
    #             )
    #         if _adapter_model_path is not None and os.path.isfile(_adapter_model_path):
    #             with open(_adapter_model_path, "r", encoding="utf-8") as f:
    #                 _adapter_model_path = pretrained_model_name_or_path
    #                 pretrained_model_name_or_path = json.load(f)["base_model_name_or_path"]
    #     else:
    #         _adapter_model_path = None

    #     # change device_map into a map if we passed an int, a str or a torch.device
    #     if isinstance(device_map, torch.device):
    #         device_map = {"": device_map}
    #     elif isinstance(device_map, str) and device_map not in ["auto", "balanced", "balanced_low_0", "sequential"]:
    #         try:
    #             device_map = {"": torch.device(device_map)}
    #         except RuntimeError:
    #             raise ValueError(
    #                 "When passing device_map as a string, the value needs to be a device name (e.g. cpu, cuda:0) or "
    #                 f"'auto', 'balanced', 'balanced_low_0', 'sequential' but found {device_map}."
    #             )
    #     elif isinstance(device_map, int):
    #         if device_map < 0:
    #             raise ValueError(
    #                 "You can't pass device_map as a negative int. If you want to put the model on the cpu, pass device_map = 'cpu' "
    #             )
    #         else:
    #             device_map = {"": device_map}

    #     if device_map is not None:
    #         if low_cpu_mem_usage is None:
    #             low_cpu_mem_usage = True
    #         elif not low_cpu_mem_usage:
    #             raise ValueError("Passing along a `device_map` or a `tp_plan` requires `low_cpu_mem_usage=True`")

    #     if low_cpu_mem_usage:
    #         if is_deepspeed_zero3_enabled():
    #             raise ValueError(
    #                 "DeepSpeed Zero-3 is not compatible with `low_cpu_mem_usage=True` or with passing a `device_map`."
    #             )
    #         elif not is_accelerate_available():
    #             raise ImportError(
    #                 f"Using `low_cpu_mem_usage=True`, a `device_map` or a `tp_plan` requires Accelerate: `pip install 'accelerate>={ACCELERATE_MIN_VERSION}'`"
    #             )

    #     # handling bnb config from kwargs, remove after `load_in_{4/8}bit` deprecation.
    #     if load_in_4bit or load_in_8bit:
    #         if quantization_config is not None:
    #             raise ValueError(
    #                 "You can't pass `load_in_4bit`or `load_in_8bit` as a kwarg when passing "
    #                 "`quantization_config` argument at the same time."
    #             )

    #         # preparing BitsAndBytesConfig from kwargs
    #         config_dict = {k: v for k, v in kwargs.items() if k in inspect.signature(BitsAndBytesConfig).parameters}
    #         config_dict = {**config_dict, "load_in_4bit": load_in_4bit, "load_in_8bit": load_in_8bit}
    #         quantization_config, kwargs = BitsAndBytesConfig.from_dict(
    #             config_dict=config_dict, return_unused_kwargs=True, **kwargs
    #         )
    #         logger.warning(
    #             "The `load_in_4bit` and `load_in_8bit` arguments are deprecated and will be removed in the future versions. "
    #             "Please, pass a `BitsAndBytesConfig` object in `quantization_config` argument instead."
    #         )

    #     from_pt = not (from_tf | from_flax)

    #     user_agent = {"file_type": "model", "framework": "pytorch", "from_auto_class": from_auto_class}
    #     if from_pipeline is not None:
    #         user_agent["using_pipeline"] = from_pipeline

    #     if is_offline_mode() and not local_files_only:
    #         logger.info("Offline mode: forcing local_files_only=True")
    #         local_files_only = True

    #     # Load config if we don't provide a configuration
    #     if not isinstance(config, PretrainedConfig):
    #         config_path = config if config is not None else pretrained_model_name_or_path
    #         config, model_kwargs = cls.config_class.from_pretrained(
    #             config_path,
    #             cache_dir=cache_dir,
    #             return_unused_kwargs=True,
    #             force_download=force_download,
    #             proxies=proxies,
    #             local_files_only=local_files_only,
    #             token=token,
    #             revision=revision,
    #             subfolder=subfolder,
    #             gguf_file=gguf_file,
    #             _from_auto=from_auto_class,
    #             _from_pipeline=from_pipeline,
    #             **kwargs,
    #         )
    #         if "gguf_file" in model_kwargs:
    #             model_kwargs.pop("gguf_file")
    #     else:
    #         # In case one passes a config to `from_pretrained` + "attn_implementation"
    #         # override the `_attn_implementation` attribute to `attn_implementation` of the kwargs
    #         # Please see: https://github.com/huggingface/transformers/issues/28038

    #         # Overwrite `config._attn_implementation` by the one from the kwargs --> in auto-factory
    #         # we pop attn_implementation from the kwargs but this handles the case where users
    #         # passes manually the config to `from_pretrained`.
    #         config = copy.deepcopy(config)

    #         kwarg_attn_imp = kwargs.pop("attn_implementation", None)
    #         if kwarg_attn_imp is not None:
    #             config._attn_implementation = kwarg_attn_imp

    #         model_kwargs = kwargs

    #     pre_quantized = hasattr(config, "quantization_config")
    #     if pre_quantized and not AutoHfQuantizer.supports_quant_method(config.quantization_config):
    #         pre_quantized = False

    #     if pre_quantized or quantization_config is not None:
    #         if pre_quantized:
    #             config.quantization_config = AutoHfQuantizer.merge_quantization_configs(
    #                 config.quantization_config, quantization_config
    #             )
    #         else:
    #             config.quantization_config = quantization_config

    #         hf_quantizer = AutoHfQuantizer.from_config(
    #             config.quantization_config,
    #             pre_quantized=pre_quantized,
    #         )
    #     else:
    #         hf_quantizer = None

    #     if hf_quantizer is not None:
    #         hf_quantizer.validate_environment(
    #             torch_dtype=torch_dtype,
    #             from_tf=from_tf,
    #             from_flax=from_flax,
    #             device_map=device_map,
    #             weights_only=weights_only,
    #         )
    #         torch_dtype = hf_quantizer.update_torch_dtype(torch_dtype)
    #         device_map = hf_quantizer.update_device_map(device_map)

    #         # In order to ensure popular quantization methods are supported. Can be disable with `disable_telemetry`
    #         if hasattr(hf_quantizer.quantization_config.quant_method, "value"):
    #             user_agent["quant"] = hf_quantizer.quantization_config.quant_method.value
    #         else:
    #             user_agent["quant"] = hf_quantizer.quantization_config.quant_method
    #         # Force-set to `True` for more mem efficiency
    #         if low_cpu_mem_usage is None:
    #             low_cpu_mem_usage = True
    #             logger.warning("`low_cpu_mem_usage` was None, now default to True since model is quantized.")

    #     if gguf_file is not None and hf_quantizer is not None:
    #         raise ValueError(
    #             "You cannot combine Quantization and loading a model from a GGUF file, try again by making sure you did not passed a `quantization_config` or that you did not load a quantized model from the Hub."
    #         )

    #     checkpoint_files, sharded_metadata = _get_resolved_checkpoint_files(
    #         pretrained_model_name_or_path=pretrained_model_name_or_path,
    #         subfolder=subfolder,
    #         variant=variant,
    #         gguf_file=gguf_file,
    #         from_tf=from_tf,
    #         from_flax=from_flax,
    #         use_safetensors=use_safetensors,
    #         cache_dir=cache_dir,
    #         force_download=force_download,
    #         proxies=proxies,
    #         local_files_only=local_files_only,
    #         token=token,
    #         user_agent=user_agent,
    #         revision=revision,
    #         commit_hash=commit_hash,
    #     )

    #     is_sharded = sharded_metadata is not None
    #     is_quantized = hf_quantizer is not None
    #     is_from_file = pretrained_model_name_or_path is not None or gguf_file is not None

    #     if (
    #         is_safetensors_available()
    #         and is_from_file
    #         and not is_sharded
    #         and checkpoint_files[0].endswith(".safetensors")
    #     ):
    #         with safe_open(checkpoint_files[0], framework="pt") as f:
    #             metadata = f.metadata()

    #         if metadata is None:
    #             # Assume it's a pytorch checkpoint (introduced for timm checkpoints)
    #             pass
    #         elif metadata.get("format") == "pt":
    #             pass
    #         elif metadata.get("format") == "tf":
    #             from_tf = True
    #             logger.info("A TensorFlow safetensors file is being loaded in a PyTorch model.")
    #         elif metadata.get("format") == "flax":
    #             from_flax = True
    #             logger.info("A Flax safetensors file is being loaded in a PyTorch model.")
    #         elif metadata.get("format") == "mlx":
    #             # This is a mlx file, we assume weights are compatible with pt
    #             pass
    #         else:
    #             raise ValueError(
    #                 f"Incompatible safetensors file. File metadata is not ['pt', 'tf', 'flax', 'mlx'] but {metadata.get('format')}"
    #             )

    #     from_pt = not (from_tf | from_flax)

    #     if from_pt:
    #         if gguf_file:
    #             from .modeling_gguf_pytorch_utils import load_gguf_checkpoint

    #             # we need a dummy model to get the state_dict - for this reason, we keep the state_dict as if it was
    #             # passed directly as a kwarg from now on
    #             with torch.device("meta"):
    #                 dummy_model = cls(config)
    #             state_dict = load_gguf_checkpoint(checkpoint_files[0], return_tensors=True, model_to_load=dummy_model)[
    #                 "tensors"
    #             ]
    #             # Force it if is not already the case
    #             low_cpu_mem_usage = True

    #         # Find the correct dtype based on current state
    #         config, torch_dtype, dtype_orig = _get_torch_dtype(
    #             cls, torch_dtype, checkpoint_files, config, sharded_metadata, state_dict, weights_only
    #         )

    #     config.name_or_path = pretrained_model_name_or_path

    #     # Instantiate model.
    #     model_init_context = cls.get_init_context(_fast_init, is_quantized, _is_ds_init_called, low_cpu_mem_usage)

    #     config = copy.deepcopy(config)  # We do not want to modify the config inplace in from_pretrained.
    #     if not getattr(config, "_attn_implementation_autoset", False):
    #         config = cls._autoset_attn_implementation(
    #             config, use_flash_attention_2=use_flash_attention_2, torch_dtype=torch_dtype, device_map=device_map
    #         )

    #     with ContextManagers(model_init_context):
    #         # Let's make sure we don't run the init function of buffer modules
    #         model = cls(config, *model_args, **model_kwargs)

    #     # Make sure to tie the weights correctly
    #     model.tie_weights()

    #     # Last check for tp
    #     if device_mesh is not None and not model.supports_tp_plan:
    #         if config.base_model_tp_plan is None and config.get_text_config().base_model_tp_plan is None:
    #             raise NotImplementedError("This model does not have a tensor parallel plan.")

    #     # make sure we use the model's config since the __init__ call might have copied it
    #     config = model.config

    #     # Find fp32 modules if needed
    #     keep_in_fp32_modules = None
    #     if model._keep_in_fp32_modules is not None:
    #         if is_accelerate_available() and not is_deepspeed_zero3_enabled():
    #             low_cpu_mem_usage = True
    #         keep_in_fp32_modules = model._keep_in_fp32_modules if len(model._keep_in_fp32_modules) > 0 else None

    #     if hf_quantizer is not None:
    #         hf_quantizer.preprocess_model(
    #             model=model, device_map=device_map, keep_in_fp32_modules=keep_in_fp32_modules
    #         )

    #         # We store the original dtype for quantized models as we cannot easily retrieve it
    #         # once the weights have been quantized
    #         # Note that once you have loaded a quantized model, you can't change its dtype so this will
    #         # remain a single source of truth
    #         config._pre_quantization_dtype = torch_dtype

    #     # Prepare the full device map
    #     if device_map is not None:
    #         device_map = _get_device_map(
    #             model, device_map, max_memory, hf_quantizer, torch_dtype, keep_in_fp32_modules
    #         )

    #     # Finalize model weight initialization
    #     if from_tf:
    #         model, loading_info = cls._load_from_tf(model, config, checkpoint_files)
    #     elif from_flax:
    #         model = cls._load_from_flax(model, checkpoint_files)
    #     elif from_pt:
    #         # restore default dtype
    #         if dtype_orig is not None:
    #             torch.set_default_dtype(dtype_orig)

    #         (
    #             model,
    #             missing_keys,
    #             unexpected_keys,
    #             mismatched_keys,
    #             offload_index,
    #             error_msgs,
    #         ) = cls._load_pretrained_model(
    #             model,
    #             state_dict,
    #             checkpoint_files,
    #             pretrained_model_name_or_path,
    #             ignore_mismatched_sizes=ignore_mismatched_sizes,
    #             sharded_metadata=sharded_metadata,
    #             low_cpu_mem_usage=low_cpu_mem_usage,
    #             device_map=device_map,
    #             disk_offload_folder=offload_folder,
    #             offload_state_dict=offload_state_dict,
    #             dtype=torch_dtype,
    #             hf_quantizer=hf_quantizer,
    #             keep_in_fp32_modules=keep_in_fp32_modules,
    #             device_mesh=device_mesh,
    #             key_mapping=key_mapping,
    #             weights_only=weights_only,
    #             _fast_init=_fast_init,
    #         )

    #     # make sure token embedding weights are still tied if needed
    #     model.tie_weights()

    #     # Set model in evaluation mode to deactivate DropOut modules by default
    #     model.eval()

    #     # If it is a model with generation capabilities, attempt to load the generation config
    #     if model.can_generate() and generation_config is not None:
    #         logger.info("The user-defined `generation_config` will be used to override the default generation config.")
    #         model.generation_config = model.generation_config.from_dict(generation_config.to_dict())
    #     elif model.can_generate() and pretrained_model_name_or_path is not None:
    #         try:
    #             model.generation_config = GenerationConfig.from_pretrained(
    #                 pretrained_model_name_or_path,
    #                 cache_dir=cache_dir,
    #                 force_download=force_download,
    #                 proxies=proxies,
    #                 local_files_only=local_files_only,
    #                 token=token,
    #                 revision=revision,
    #                 subfolder=subfolder,
    #                 _from_auto=from_auto_class,
    #                 _from_pipeline=from_pipeline,
    #                 **kwargs,
    #             )
    #         except OSError:
    #             logger.info(
    #                 "Generation config file not found, using a generation config created from the model config."
    #             )
    #             pass

    #     # Dispatch model with hooks on all devices if necessary (not needed with a tp_plan, so we skip it as it slightly
    #     # harm performances)
    #     if device_map is not None and device_mesh is None:
    #         device_map_kwargs = {
    #             "device_map": device_map,
    #             "offload_dir": offload_folder,
    #             "offload_index": offload_index,
    #             "offload_buffers": offload_buffers,
    #         }
    #         if "skip_keys" in inspect.signature(dispatch_model).parameters:
    #             device_map_kwargs["skip_keys"] = model._skip_keys_device_placement
    #         # For HQQ method we force-set the hooks for single GPU envs
    #         if (
    #             "force_hooks" in inspect.signature(dispatch_model).parameters
    #             and hf_quantizer is not None
    #             and hf_quantizer.quantization_config.quant_method == QuantizationMethod.HQQ
    #         ):
    #             device_map_kwargs["force_hooks"] = True
    #         if (
    #             hf_quantizer is not None
    #             and hf_quantizer.quantization_config.quant_method == QuantizationMethod.FBGEMM_FP8
    #             and isinstance(device_map, dict)
    #             and ("cpu" in device_map.values() or "disk" in device_map.values())
    #         ):
    #             device_map_kwargs["offload_buffers"] = True

    #         if not is_fsdp_enabled() and not is_deepspeed_zero3_enabled():
    #             dispatch_model(model, **device_map_kwargs)

    #     if hf_quantizer is not None:
    #         hf_quantizer.postprocess_model(model, config=config)
    #         model.hf_quantizer = hf_quantizer

    #     if _adapter_model_path is not None:
    #         model.load_adapter(
    #             _adapter_model_path,
    #             adapter_name=adapter_name,
    #             token=token,
    #             adapter_kwargs=adapter_kwargs,
    #         )

    #     if output_loading_info:
    #         if from_pt:
    #             loading_info = {
    #                 "missing_keys": missing_keys,
    #                 "unexpected_keys": unexpected_keys,
    #                 "mismatched_keys": mismatched_keys,
    #                 "error_msgs": error_msgs,
    #             }
    #         elif from_flax:
    #             loading_info = None
    #         return model, loading_info

    #     return model
        #     config (`Union[PretrainedConfig, str, os.PathLike]`, *optional*):
        #         Can be either:

        #             - an instance of a class derived from [`PretrainedConfig`],
        #             - a string or path valid as input to [`~PretrainedConfig.from_pretrained`].

        #         Configuration for the model to use instead of an automatically loaded configuration. Configuration can
        #         be automatically loaded when:

        #             - The model is a model provided by the library (loaded with the *model id* string of a pretrained
        #               model).
        #             - The model was saved using [`~PreTrainedModel.save_pretrained`] and is reloaded by supplying the
        #               save directory.
        #             - The model is loaded by supplying a local directory as `pretrained_model_name_or_path` and a
        #               configuration JSON file named *config.json* is found in the directory.
        #     state_dict (`Dict[str, torch.Tensor]`, *optional*):
        #         A state dictionary to use instead of a state dictionary loaded from saved weights file.

        #         This option can be used if you want to create a model from a pretrained configuration but load your own
        #         weights. In this case though, you should check if using [`~PreTrainedModel.save_pretrained`] and
        #         [`~PreTrainedModel.from_pretrained`] is not a simpler option.
        #     cache_dir (`Union[str, os.PathLike]`, *optional*):
        #         Path to a directory in which a downloaded pretrained model configuration should be cached if the
        #         standard cache should not be used.
        #     from_tf (`bool`, *optional*, defaults to `False`):
        #         Load the model weights from a TensorFlow checkpoint save file (see docstring of
        #         `pretrained_model_name_or_path` argument).
        #     from_flax (`bool`, *optional*, defaults to `False`):
        #         Load the model weights from a Flax checkpoint save file (see docstring of
        #         `pretrained_model_name_or_path` argument).
        #     ignore_mismatched_sizes (`bool`, *optional*, defaults to `False`):
        #         Whether or not to raise an error if some of the weights from the checkpoint do not have the same size
        #         as the weights of the model (if for instance, you are instantiating a model with 10 labels from a
        #         checkpoint with 3 labels).
        #     force_download (`bool`, *optional*, defaults to `False`):
        #         Whether or not to force the (re-)download of the model weights and configuration files, overriding the
        #         cached versions if they exist.
        #     resume_download:
        #         Deprecated and ignored. All downloads are now resumed by default when possible.
        #         Will be removed in v5 of Transformers.
        #     proxies (`Dict[str, str]`, *optional*):
        #         A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
        #         'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
        #     output_loading_info(`bool`, *optional*, defaults to `False`):
        #         Whether ot not to also return a dictionary containing missing keys, unexpected keys and error messages.
        #     local_files_only(`bool`, *optional*, defaults to `False`):
        #         Whether or not to only look at local files (i.e., do not try to download the model).
        #     token (`str` or `bool`, *optional*):
        #         The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
        #         the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).
        #     revision (`str`, *optional*, defaults to `"main"`):
        #         The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
        #         git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
        #         identifier allowed by git.

        #         <Tip>

        #         To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>"`.

        #         </Tip>
        #     _fast_init(`bool`, *optional*, defaults to `True`):
        #         Whether or not to disable fast initialization.

        #         <Tip warning={true}>

        #         One should only disable *_fast_init* to ensure backwards compatibility with `transformers.__version__ <
        #         4.6.0` for seeded model initialization. This argument will be removed at the next major version. See
        #         [pull request 11471](https://github.com/huggingface/transformers/pull/11471) for more information.

        #         </Tip>
        #     attn_implementation (`str`, *optional*):
        #         The attention implementation to use in the model (if relevant). Can be any of `"eager"` (manual implementation of the attention), `"sdpa"` (using [`F.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention.html)), or `"flash_attention_2"` (using [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)). By default, if available, SDPA will be used for torch>=2.1.1. The default is otherwise the manual `"eager"` implementation.

        #     > Parameters for big model inference

        #     low_cpu_mem_usage(`bool`, *optional*):
        #         Tries not to use more than 1x model size in CPU memory (including peak memory) while loading the model.
        #         Generally should be combined with a `device_map` (such as `"auto"`) for best results.
        #         This is an experimental feature and a subject to change at any moment.
        #         </Tip>
        #             If the model weights are in the same precision as the model loaded in, `low_cpu_mem_usage` (without
        #             `device_map`) is redundant and will not provide any benefit in regards to CPU memory usage. However,
        #             this should still be enabled if you are passing in a `device_map`.
        #         </Tip>
        #     torch_dtype (`str` or `torch.dtype`, *optional*):
        #         Override the default `torch.dtype` and load the model under a specific `dtype`. The different options
        #         are:

        #         1. `torch.float16` or `torch.bfloat16` or `torch.float`: load in a specified
        #           `dtype`, ignoring the model's `config.torch_dtype` if one exists. If not specified
        #           - the model will get loaded in `torch.float` (fp32).

        #         2. `"auto"` - A `torch_dtype` entry in the `config.json` file of the model will be
        #           attempted to be used. If this entry isn't found then next check the `dtype` of the first weight in
        #           the checkpoint that's of a floating point type and use that as `dtype`. This will load the model
        #           using the `dtype` it was saved in at the end of the training. It can't be used as an indicator of how
        #           the model was trained. Since it could be trained in one of half precision dtypes, but saved in fp32.

        #         3. A string that is a valid `torch.dtype`. E.g. "float32" loads the model in `torch.float32`, "float16" loads in `torch.float16` etc.

        #         <Tip>

        #         For some models the `dtype` they were trained in is unknown - you may try to check the model's paper or
        #         reach out to the authors and ask them to add this information to the model's card and to insert the
        #         `torch_dtype` entry in `config.json` on the hub.

        #         </Tip>

        #     device_map (`str` or `Dict[str, Union[int, str, torch.device]]` or `int` or `torch.device`, *optional*):
        #         A map that specifies where each submodule should go. It doesn't need to be refined to each
        #         parameter/buffer name, once a given module name is inside, every submodule of it will be sent to the
        #         same device. If we only pass the device (*e.g.*, `"cpu"`, `"cuda:1"`, `"mps"`, or a GPU ordinal rank
        #         like `1`) on which the model will be allocated, the device map will map the entire model to this
        #         device. Passing `device_map = 0` means put the whole model on GPU 0.

        #         To have Accelerate compute the most optimized `device_map` automatically, set `device_map="auto"`. For
        #         more information about each option see [designing a device
        #         map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
        #     max_memory (`Dict`, *optional*):
        #         A dictionary device identifier to maximum memory if using `device_map`. Will default to the maximum memory available for each
        #         GPU and the available CPU RAM if unset.
        #     tp_plan (`str`, *optional*):
        #         A torch tensor parallel plan, see [here](https://pytorch.org/tutorials/intermediate/TP_tutorial.html). Currently, it only accepts
        #         `tp_plan="auto"` to use predefined plan based on the model. Note that if you use it, you should launch your script accordingly with
        #         `torchrun [args] script.py`. This will be much faster than using a `device_map`, but has limitations.
        #     offload_folder (`str` or `os.PathLike`, *optional*):
        #         If the `device_map` contains any value `"disk"`, the folder where we will offload weights.
        #     offload_state_dict (`bool`, *optional*):
        #         If `True`, will temporarily offload the CPU state dict to the hard drive to avoid getting out of CPU
        #         RAM if the weight of the CPU state dict + the biggest shard of the checkpoint does not fit. Defaults to
        #         `True` when there is some disk offload.
        #     offload_buffers (`bool`, *optional*):
        #         Whether or not to offload the buffers with the model parameters.
        #     quantization_config (`Union[QuantizationConfigMixin,Dict]`, *optional*):
        #         A dictionary of configuration parameters or a QuantizationConfigMixin object for quantization (e.g
        #         bitsandbytes, gptq). There may be other quantization-related kwargs, including `load_in_4bit` and
        #         `load_in_8bit`, which are parsed by QuantizationConfigParser. Supported only for bitsandbytes
        #         quantizations and not preferred. consider inserting all such arguments into quantization_config
        #         instead.
        #     subfolder (`str`, *optional*, defaults to `""`):
        #         In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
        #         specify the folder name here.
        #     variant (`str`, *optional*):
        #         If specified load weights from `variant` filename, *e.g.* pytorch_model.<variant>.bin. `variant` is
        #         ignored when using `from_tf` or `from_flax`.
        #     use_safetensors (`bool`, *optional*, defaults to `None`):
        #         Whether or not to use `safetensors` checkpoints. Defaults to `None`. If not specified and `safetensors`
        #         is not installed, it will be set to `False`.
        #     weights_only (`bool`, *optional*, defaults to `True`):
        #         Indicates whether unpickler should be restricted to loading only tensors, primitive types,
        #         dictionaries and any types added via torch.serialization.add_safe_globals().
        #         When set to False, we can load wrapper tensor subclass weights.
        #     key_mapping (`Dict[str, str], *optional*):
        #         A potential mapping of the weight names if using a model on the Hub which is compatible to a Transformers
        #         architecture, but was not converted accordingly.
        #     kwargs (remaining dictionary of keyword arguments, *optional*):
        #         Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
        #         `output_attentions=True`). Behaves differently depending on whether a `config` is provided or
        #         automatically loaded:

        #             - If a configuration is provided with `config`, `**kwargs` will be directly passed to the
        #               underlying model's `__init__` method (we assume all relevant updates to the configuration have
        #               already been done)
        #             - If a configuration is not provided, `kwargs` will be first passed to the configuration class
        #               initialization function ([`~PretrainedConfig.from_pretrained`]). Each key of `kwargs` that
        #               corresponds to a configuration attribute will be used to override said attribute with the
        #               supplied `kwargs` value. Remaining keys that do not correspond to any configuration attribute
        #               will be passed to the underlying model's `__init__` function.

        # <Tip>

        # Activate the special ["offline-mode"](https://huggingface.co/transformers/installation.html#offline-mode) to
        # use this method in a firewalled environment.

        # </Tip>

        # Examples:

        # ```python
        # >>> from transformers import BertConfig, BertModel

        # >>> # Download model and configuration from huggingface.co and cache.
        # >>> model = BertModel.from_pretrained("google-bert/bert-base-uncased")
        # >>> # Model was saved using *save_pretrained('./test/saved_model/')* (for example purposes, not runnable).
        # >>> model = BertModel.from_pretrained("./test/saved_model/")
        # >>> # Update configuration during loading.
        # >>> model = BertModel.from_pretrained("google-bert/bert-base-uncased", output_attentions=True)
        # >>> assert model.config.output_attentions == True
        # >>> # Loading from a TF checkpoint file instead of a PyTorch model (slower, for example purposes, not runnable).
        # >>> config = BertConfig.from_json_file("./tf_model/my_tf_model_config.json")
        # >>> model = BertModel.from_pretrained("./tf_model/my_tf_checkpoint.ckpt.index", from_tf=True, config=config)
        # >>> # Loading from a Flax checkpoint file instead of a PyTorch model (slower)
        # >>> model = BertModel.from_pretrained("google-bert/bert-base-uncased", from_flax=True)
        # ```

        # * `low_cpu_mem_usage` algorithm:

        # This is an experimental function that loads the model using ~1x model size CPU memory

        # Here is how it works:

        # 1. save which state_dict keys we have
        # 2. drop state_dict before the model is created, since the latter takes 1x model size CPU memory
        # 3. after the model has been instantiated switch to the meta device all params/buffers that
        # are going to be replaced from the loaded state_dict
        # 4. load state_dict 2nd time
        # 5. replace the params/buffers from the state_dict

        # Currently, it can't handle deepspeed ZeRO stage 3 and ignores loading errors

        # """
        # state_dict = kwargs.pop("state_dict", None)
        # from_tf = kwargs.pop("from_tf", False)
        # from_flax = kwargs.pop("from_flax", False)
        # _ = kwargs.pop("resume_download", None)
        # proxies = kwargs.pop("proxies", None)
        # output_loading_info = kwargs.pop("output_loading_info", False)
        # use_auth_token = kwargs.pop("use_auth_token", None)
        # _ = kwargs.pop("trust_remote_code", None)
        # _ = kwargs.pop("mirror", None)
        # from_pipeline = kwargs.pop("_from_pipeline", None)
        # from_auto_class = kwargs.pop("_from_auto", False)
        # _fast_init = kwargs.pop("_fast_init", True)
        # torch_dtype = kwargs.pop("torch_dtype", None)
        # low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", None)
        # device_map = kwargs.pop("device_map", None)
        # max_memory = kwargs.pop("max_memory", None)
        # offload_folder = kwargs.pop("offload_folder", None)
        # offload_state_dict = kwargs.pop("offload_state_dict", False)
        # offload_buffers = kwargs.pop("offload_buffers", False)
        # load_in_8bit = kwargs.pop("load_in_8bit", False)
        # load_in_4bit = kwargs.pop("load_in_4bit", False)
        # quantization_config = kwargs.pop("quantization_config", None)
        # subfolder = kwargs.pop("subfolder", "")
        # commit_hash = kwargs.pop("_commit_hash", None)
        # variant = kwargs.pop("variant", None)
        # adapter_kwargs = kwargs.pop("adapter_kwargs", {})
        # adapter_name = kwargs.pop("adapter_name", "default")
        # use_flash_attention_2 = kwargs.pop("use_flash_attention_2", False)
        # generation_config = kwargs.pop("generation_config", None)
        # gguf_file = kwargs.pop("gguf_file", None)
        # tp_plan = kwargs.pop("tp_plan", None)
        # key_mapping = kwargs.pop("key_mapping", None)

        # if state_dict is not None and (pretrained_model_name_or_path is not None or gguf_file is not None):
        #     raise ValueError(
        #         "`state_dict` cannot be passed together with a model name or a `gguf_file`. Use one of the two loading strategies."
        #     )

        # if tp_plan is not None and tp_plan != "auto":
        #     # TODO: we can relax this check when we support taking tp_plan from a json file, for example.
        #     raise ValueError(f"tp_plan supports 'auto' only for now but got {tp_plan}.")
        # if tp_plan is not None and device_map is not None:
        #     raise ValueError(
        #         "`tp_plan` and `device_map` are mutually exclusive. Choose either one for parallelization."
        #     )

        # # If torchrun was used, make sure to TP by default. This way people don't need to change tp or device map
        # if device_map == "auto" and tp_plan is None and int(os.environ.get("WORLD_SIZE", 0)):
        #     tp_plan = "auto"  # device_map = "auto" in torchrun equivalent to TP plan = AUTO!
        #     device_map = None

        # # We need to correctly dispatch the model on the current process device. The easiest way for this is to use a simple
        # # `device_map` pointing to the correct device
        # device_mesh = None
        # if tp_plan is not None:
        #     if not is_torch_greater_or_equal("2.5"):
        #         raise EnvironmentError("tensor parallel is only supported for `torch>=2.5`.")
        #     if not torch.distributed.is_initialized():
        #         try:
        #             logger.warning("Tensor Parallel requires torch.distributed to be initialized first.")
        #             rank = int(os.environ["RANK"])
        #             world_size = int(os.environ["WORLD_SIZE"])
        #             torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
        #             torch.cuda.set_device(rank)
        #         except Exception as e:
        #             raise EnvironmentError(
        #                 "We tried to initialize torch.distributed for you, but it failed, make"
        #                 "sure you init torch distributed in your script to use `tp_plan='auto'`"
        #             ) from e

        #     # Detect the accelerator on the machine. If no accelerator is available, it returns CPU.
        #     device_type = torch._C._get_accelerator().type
        #     device_module = torch.get_device_module(device_type)
        #     # Get device with index assuming equal number of devices per host
        #     tp_device = torch.device(device_type, torch.distributed.get_rank() % device_module.device_count())
        #     # This is the easiest way to dispatch to the current process device
        #     device_map = tp_device

        #     # Assuming sharding the model onto the world
        #     world_size = torch.distributed.get_world_size()
        #     device_mesh = torch.distributed.init_device_mesh(tp_device.type, (world_size,))

        # if is_fsdp_enabled():
        #     low_cpu_mem_usage = True

        # if use_auth_token is not None:
        #     warnings.warn(
        #         "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
        #         FutureWarning,
        #     )
        #     if token is not None:
        #         raise ValueError(
        #             "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
        #         )
        #     token = use_auth_token

        # if token is not None and adapter_kwargs is not None and "token" not in adapter_kwargs:
        #     adapter_kwargs["token"] = token

        # if use_safetensors is None and not is_safetensors_available():
        #     use_safetensors = False

        # if gguf_file is not None and not is_accelerate_available():
        #     raise ValueError("accelerate is required when loading a GGUF file `pip install accelerate`.")

        # if commit_hash is None:
        #     if not isinstance(config, PretrainedConfig):
        #         # We make a call to the config file first (which may be absent) to get the commit hash as soon as possible
        #         resolved_config_file = cached_file(
        #             pretrained_model_name_or_path,
        #             CONFIG_NAME,
        #             cache_dir=cache_dir,
        #             force_download=force_download,
        #             proxies=proxies,
        #             local_files_only=local_files_only,
        #             token=token,
        #             revision=revision,
        #             subfolder=subfolder,
        #             _raise_exceptions_for_gated_repo=False,
        #             _raise_exceptions_for_missing_entries=False,
        #             _raise_exceptions_for_connection_errors=False,
        #         )
        #         commit_hash = extract_commit_hash(resolved_config_file, commit_hash)
        #     else:
        #         commit_hash = getattr(config, "_commit_hash", None)

        # if is_peft_available():
        #     _adapter_model_path = adapter_kwargs.pop("_adapter_model_path", None)

        #     if _adapter_model_path is None:
        #         _adapter_model_path = find_adapter_config_file(
        #             pretrained_model_name_or_path,
        #             cache_dir=cache_dir,
        #             force_download=force_download,
        #             proxies=proxies,
        #             local_files_only=local_files_only,
        #             _commit_hash=commit_hash,
        #             **adapter_kwargs,
        #         )
        #     if _adapter_model_path is not None and os.path.isfile(_adapter_model_path):
        #         with open(_adapter_model_path, "r", encoding="utf-8") as f:
        #             _adapter_model_path = pretrained_model_name_or_path
        #             pretrained_model_name_or_path = json.load(f)["base_model_name_or_path"]
        # else:
        #     _adapter_model_path = None

        # # change device_map into a map if we passed an int, a str or a torch.device
        # if isinstance(device_map, torch.device):
        #     device_map = {"": device_map}
        # elif isinstance(device_map, str) and device_map not in ["auto", "balanced", "balanced_low_0", "sequential"]:
        #     try:
        #         device_map = {"": torch.device(device_map)}
        #     except RuntimeError:
        #         raise ValueError(
        #             "When passing device_map as a string, the value needs to be a device name (e.g. cpu, cuda:0) or "
        #             f"'auto', 'balanced', 'balanced_low_0', 'sequential' but found {device_map}."
        #         )
        # elif isinstance(device_map, int):
        #     if device_map < 0:
        #         raise ValueError(
        #             "You can't pass device_map as a negative int. If you want to put the model on the cpu, pass device_map = 'cpu' "
        #         )
        #     else:
        #         device_map = {"": device_map}

        # if device_map is not None:
        #     if low_cpu_mem_usage is None:
        #         low_cpu_mem_usage = True
        #     elif not low_cpu_mem_usage:
        #         raise ValueError("Passing along a `device_map` or a `tp_plan` requires `low_cpu_mem_usage=True`")

        # if low_cpu_mem_usage:
        #     if is_deepspeed_zero3_enabled():
        #         raise ValueError(
        #             "DeepSpeed Zero-3 is not compatible with `low_cpu_mem_usage=True` or with passing a `device_map`."
        #         )
        #     elif not is_accelerate_available():
        #         raise ImportError(
        #             f"Using `low_cpu_mem_usage=True`, a `device_map` or a `tp_plan` requires Accelerate: `pip install 'accelerate>={ACCELERATE_MIN_VERSION}'`"
        #         )

        # # handling bnb config from kwargs, remove after `load_in_{4/8}bit` deprecation.
        # if load_in_4bit or load_in_8bit:
        #     if quantization_config is not None:
        #         raise ValueError(
        #             "You can't pass `load_in_4bit`or `load_in_8bit` as a kwarg when passing "
        #             "`quantization_config` argument at the same time."
        #         )

        #     # preparing BitsAndBytesConfig from kwargs
        #     config_dict = {k: v for k, v in kwargs.items() if k in inspect.signature(BitsAndBytesConfig).parameters}
        #     config_dict = {**config_dict, "load_in_4bit": load_in_4bit, "load_in_8bit": load_in_8bit}
        #     quantization_config, kwargs = BitsAndBytesConfig.from_dict(
        #         config_dict=config_dict, return_unused_kwargs=True, **kwargs
        #     )
        #     logger.warning(
        #         "The `load_in_4bit` and `load_in_8bit` arguments are deprecated and will be removed in the future versions. "
        #         "Please, pass a `BitsAndBytesConfig` object in `quantization_config` argument instead."
        #     )

        # from_pt = not (from_tf | from_flax)

        # user_agent = {"file_type": "model", "framework": "pytorch", "from_auto_class": from_auto_class}
        # if from_pipeline is not None:
        #     user_agent["using_pipeline"] = from_pipeline

        # if is_offline_mode() and not local_files_only:
        #     logger.info("Offline mode: forcing local_files_only=True")
        #     local_files_only = True

        # # Load config if we don't provide a configuration
        # if not isinstance(config, PretrainedConfig):
        #     config_path = config if config is not None else pretrained_model_name_or_path
        #     config, model_kwargs = cls.config_class.from_pretrained(
        #         config_path,
        #         cache_dir=cache_dir,
        #         return_unused_kwargs=True,
        #         force_download=force_download,
        #         proxies=proxies,
        #         local_files_only=local_files_only,
        #         token=token,
        #         revision=revision,
        #         subfolder=subfolder,
        #         gguf_file=gguf_file,
        #         _from_auto=from_auto_class,
        #         _from_pipeline=from_pipeline,
        #         **kwargs,
        #     )
        #     if "gguf_file" in model_kwargs:
        #         model_kwargs.pop("gguf_file")
        # else:
        #     # In case one passes a config to `from_pretrained` + "attn_implementation"
        #     # override the `_attn_implementation` attribute to `attn_implementation` of the kwargs
        #     # Please see: https://github.com/huggingface/transformers/issues/28038

        #     # Overwrite `config._attn_implementation` by the one from the kwargs --> in auto-factory
        #     # we pop attn_implementation from the kwargs but this handles the case where users
        #     # passes manually the config to `from_pretrained`.
        #     config = copy.deepcopy(config)

        #     kwarg_attn_imp = kwargs.pop("attn_implementation", None)
        #     if kwarg_attn_imp is not None:
        #         config._attn_implementation = kwarg_attn_imp

        #     model_kwargs = kwargs

        # pre_quantized = hasattr(config, "quantization_config")
        # if pre_quantized and not AutoHfQuantizer.supports_quant_method(config.quantization_config):
        #     pre_quantized = False

        # if pre_quantized or quantization_config is not None:
        #     if pre_quantized:
        #         config.quantization_config = AutoHfQuantizer.merge_quantization_configs(
        #             config.quantization_config, quantization_config
        #         )
        #     else:
        #         config.quantization_config = quantization_config

        #     hf_quantizer = AutoHfQuantizer.from_config(
        #         config.quantization_config,
        #         pre_quantized=pre_quantized,
        #     )
        # else:
        #     hf_quantizer = None

        # if hf_quantizer is not None:
        #     hf_quantizer.validate_environment(
        #         torch_dtype=torch_dtype,
        #         from_tf=from_tf,
        #         from_flax=from_flax,
        #         device_map=device_map,
        #         weights_only=weights_only,
        #     )
        #     torch_dtype = hf_quantizer.update_torch_dtype(torch_dtype)
        #     device_map = hf_quantizer.update_device_map(device_map)

        #     # In order to ensure popular quantization methods are supported. Can be disable with `disable_telemetry`
        #     if hasattr(hf_quantizer.quantization_config.quant_method, "value"):
        #         user_agent["quant"] = hf_quantizer.quantization_config.quant_method.value
        #     else:
        #         user_agent["quant"] = hf_quantizer.quantization_config.quant_method
        #     # Force-set to `True` for more mem efficiency
        #     if low_cpu_mem_usage is None:
        #         low_cpu_mem_usage = True
        #         logger.warning("`low_cpu_mem_usage` was None, now default to True since model is quantized.")

        # if gguf_file is not None and hf_quantizer is not None:
        #     raise ValueError(
        #         "You cannot combine Quantization and loading a model from a GGUF file, try again by making sure you did not passed a `quantization_config` or that you did not load a quantized model from the Hub."
        #     )

        # checkpoint_files, sharded_metadata = _get_resolved_checkpoint_files(
        #     pretrained_model_name_or_path=pretrained_model_name_or_path,
        #     subfolder=subfolder,
        #     variant=variant,
        #     gguf_file=gguf_file,
        #     from_tf=from_tf,
        #     from_flax=from_flax,
        #     use_safetensors=use_safetensors,
        #     cache_dir=cache_dir,
        #     force_download=force_download,
        #     proxies=proxies,
        #     local_files_only=local_files_only,
        #     token=token,
        #     user_agent=user_agent,
        #     revision=revision,
        #     commit_hash=commit_hash,
        # )

        # is_sharded = sharded_metadata is not None
        # is_quantized = hf_quantizer is not None
        # is_from_file = pretrained_model_name_or_path is not None or gguf_file is not None

        # if (
        #     is_safetensors_available()
        #     and is_from_file
        #     and not is_sharded
        #     and checkpoint_files[0].endswith(".safetensors")
        # ):
        #     with safe_open(checkpoint_files[0], framework="pt") as f:
        #         metadata = f.metadata()

        #     if metadata is None:
        #         # Assume it's a pytorch checkpoint (introduced for timm checkpoints)
        #         pass
        #     elif metadata.get("format") == "pt":
        #         pass
        #     elif metadata.get("format") == "tf":
        #         from_tf = True
        #         logger.info("A TensorFlow safetensors file is being loaded in a PyTorch model.")
        #     elif metadata.get("format") == "flax":
        #         from_flax = True
        #         logger.info("A Flax safetensors file is being loaded in a PyTorch model.")
        #     elif metadata.get("format") == "mlx":
        #         # This is a mlx file, we assume weights are compatible with pt
        #         pass
        #     else:
        #         raise ValueError(
        #             f"Incompatible safetensors file. File metadata is not ['pt', 'tf', 'flax', 'mlx'] but {metadata.get('format')}"
        #         )

        # from_pt = not (from_tf | from_flax)

        # if from_pt:
        #     if gguf_file:
        #         from .modeling_gguf_pytorch_utils import load_gguf_checkpoint

        #         # we need a dummy model to get the state_dict - for this reason, we keep the state_dict as if it was
        #         # passed directly as a kwarg from now on
        #         with torch.device("meta"):
        #             dummy_model = cls(config)
        #         state_dict = load_gguf_checkpoint(checkpoint_files[0], return_tensors=True, model_to_load=dummy_model)[
        #             "tensors"
        #         ]
        #         # Force it if is not already the case
        #         low_cpu_mem_usage = True

        #     # Find the correct dtype based on current state
        #     config, torch_dtype, dtype_orig = _get_torch_dtype(
        #         cls, torch_dtype, checkpoint_files, config, sharded_metadata, state_dict, weights_only
        #     )

        # config.name_or_path = pretrained_model_name_or_path

        # # Instantiate model.
        # model_init_context = cls.get_init_context(_fast_init, is_quantized, _is_ds_init_called, low_cpu_mem_usage)

        # config = copy.deepcopy(config)  # We do not want to modify the config inplace in from_pretrained.
        # if not getattr(config, "_attn_implementation_autoset", False):
        #     config = cls._autoset_attn_implementation(
        #         config, use_flash_attention_2=use_flash_attention_2, torch_dtype=torch_dtype, device_map=device_map
        #     )

        # with ContextManagers(model_init_context):
        #     # Let's make sure we don't run the init function of buffer modules
        #     model = cls(config, *model_args, **model_kwargs)

        # # Make sure to tie the weights correctly
        # model.tie_weights()

        # # Last check for tp
        # if device_mesh is not None and not model.supports_tp_plan:
        #     if config.base_model_tp_plan is None and config.get_text_config().base_model_tp_plan is None:
        #         raise NotImplementedError("This model does not have a tensor parallel plan.")

        # # make sure we use the model's config since the __init__ call might have copied it
        # config = model.config

        # # Find fp32 modules if needed
        # keep_in_fp32_modules = None
        # if model._keep_in_fp32_modules is not None:
        #     if is_accelerate_available() and not is_deepspeed_zero3_enabled():
        #         low_cpu_mem_usage = True
        #     keep_in_fp32_modules = model._keep_in_fp32_modules if len(model._keep_in_fp32_modules) > 0 else None

        # if hf_quantizer is not None:
        #     hf_quantizer.preprocess_model(
        #         model=model, device_map=device_map, keep_in_fp32_modules=keep_in_fp32_modules
        #     )

        #     # We store the original dtype for quantized models as we cannot easily retrieve it
        #     # once the weights have been quantized
        #     # Note that once you have loaded a quantized model, you can't change its dtype so this will
        #     # remain a single source of truth
        #     config._pre_quantization_dtype = torch_dtype

        # # Prepare the full device map
        # if device_map is not None:
        #     device_map = _get_device_map(
        #         model, device_map, max_memory, hf_quantizer, torch_dtype, keep_in_fp32_modules
        #     )

        # # Finalize model weight initialization
        # if from_tf:
        #     model, loading_info = cls._load_from_tf(model, config, checkpoint_files)
        # elif from_flax:
        #     model = cls._load_from_flax(model, checkpoint_files)
        # elif from_pt:
        #     # restore default dtype
        #     if dtype_orig is not None:
        #         torch.set_default_dtype(dtype_orig)

        #     (
        #         model,
        #         missing_keys,
        #         unexpected_keys,
        #         mismatched_keys,
        #         offload_index,
        #         error_msgs,
        #     ) = cls._load_pretrained_model(
        #         model,
        #         state_dict,
        #         checkpoint_files,
        #         pretrained_model_name_or_path,
        #         ignore_mismatched_sizes=ignore_mismatched_sizes,
        #         sharded_metadata=sharded_metadata,
        #         low_cpu_mem_usage=low_cpu_mem_usage,
        #         device_map=device_map,
        #         disk_offload_folder=offload_folder,
        #         offload_state_dict=offload_state_dict,
        #         dtype=torch_dtype,
        #         hf_quantizer=hf_quantizer,
        #         keep_in_fp32_modules=keep_in_fp32_modules,
        #         device_mesh=device_mesh,
        #         key_mapping=key_mapping,
        #         weights_only=weights_only,
        #         _fast_init=_fast_init,
        #     )

        # # make sure token embedding weights are still tied if needed
        # model.tie_weights()

        # # Set model in evaluation mode to deactivate DropOut modules by default
        # model.eval()

        # # If it is a model with generation capabilities, attempt to load the generation config
        # if model.can_generate() and generation_config is not None:
        #     logger.info("The user-defined `generation_config` will be used to override the default generation config.")
        #     model.generation_config = model.generation_config.from_dict(generation_config.to_dict())
        # elif model.can_generate() and pretrained_model_name_or_path is not None:
        #     try:
        #         model.generation_config = GenerationConfig.from_pretrained(
        #             pretrained_model_name_or_path,
        #             cache_dir=cache_dir,
        #             force_download=force_download,
        #             proxies=proxies,
        #             local_files_only=local_files_only,
        #             token=token,
        #             revision=revision,
        #             subfolder=subfolder,
        #             _from_auto=from_auto_class,
        #             _from_pipeline=from_pipeline,
        #             **kwargs,
        #         )
        #     except OSError:
        #         logger.info(
        #             "Generation config file not found, using a generation config created from the model config."
        #         )
        #         pass

        # # Dispatch model with hooks on all devices if necessary (not needed with a tp_plan, so we skip it as it slightly
        # # harm performances)
        # if device_map is not None and device_mesh is None:
        #     device_map_kwargs = {
        #         "device_map": device_map,
        #         "offload_dir": offload_folder,
        #         "offload_index": offload_index,
        #         "offload_buffers": offload_buffers,
        #     }
        #     if "skip_keys" in inspect.signature(dispatch_model).parameters:
        #         device_map_kwargs["skip_keys"] = model._skip_keys_device_placement
        #     # For HQQ method we force-set the hooks for single GPU envs
        #     if (
        #         "force_hooks" in inspect.signature(dispatch_model).parameters
        #         and hf_quantizer is not None
        #         and hf_quantizer.quantization_config.quant_method == QuantizationMethod.HQQ
        #     ):
        #         device_map_kwargs["force_hooks"] = True
        #     if (
        #         hf_quantizer is not None
        #         and hf_quantizer.quantization_config.quant_method == QuantizationMethod.FBGEMM_FP8
        #         and isinstance(device_map, dict)
        #         and ("cpu" in device_map.values() or "disk" in device_map.values())
        #     ):
        #         device_map_kwargs["offload_buffers"] = True

        #     if not is_fsdp_enabled() and not is_deepspeed_zero3_enabled():
        #         dispatch_model(model, **device_map_kwargs)

        # if hf_quantizer is not None:
        #     hf_quantizer.postprocess_model(model, config=config)
        #     model.hf_quantizer = hf_quantizer

        # if _adapter_model_path is not None:
        #     model.load_adapter(
        #         _adapter_model_path,
        #         adapter_name=adapter_name,
        #         token=token,
        #         adapter_kwargs=adapter_kwargs,
        #     )

        # if output_loading_info:
        #     if from_pt:
        #         loading_info = {
        #             "missing_keys": missing_keys,
        #             "unexpected_keys": unexpected_keys,
        #             "mismatched_keys": mismatched_keys,
        #             "error_msgs": error_msgs,
        #         }
        #     elif from_flax:
        #         loading_info = None
        #     return model, loading_info

        # return model

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
