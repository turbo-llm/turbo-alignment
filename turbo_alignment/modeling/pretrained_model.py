# pylint: disable=line-too-long,superfluous-parens,abstract-method,raise-missing-from,no-else-raise,protected-access,unnecessary-pass,unbalanced-tuple-unpacking,unexpected-keyword-arg,attribute-defined-outside-init
# flake8: noqa
# mypy: ignore-errors
import collections
import copy
import gc
import importlib
import json
import os
import re
import warnings
from typing import Callable, Optional, Union

import torch
from packaging import version
from transformers.integrations.deepspeed import is_deepspeed_available
from transformers.modeling_utils import (
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    IS_SAGEMAKER_MP_POST_1_10,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    ContextManagers,
    PreTrainedModel,
    _add_variant,
    _find_disjoint,
    _find_identical,
    _get_tied_weight_keys,
    _is_ds_init_called,
    _is_quantized,
    create_and_tag_model_card,
    custom_object_save,
    deepspeed_config,
    find_tied_parameters,
    get_parameter_dtype,
    id_tensor_storage,
    init_empty_weights,
    is_deepspeed_zero3_enabled,
    logger,
    no_init_weights,
    restore_default_torch_dtype,
    safe_save_file,
    set_quantized_state,
    set_zero3_state,
    split_torch_state_dict_into_shards,
    unwrap_model,
)
from transformers.quantizers.base import HfQuantizer
from transformers.utils import is_safetensors_available, logging
from transformers.utils.import_utils import is_accelerate_available

if is_accelerate_available():
    accelerate_version = version.parse(importlib.metadata.version("accelerate"))
    if accelerate_version >= version.parse("0.31"):
        from accelerate.utils.modeling import get_state_dict_from_offload

if is_deepspeed_available():
    import deepspeed


class PreTrainedModelWithMPU(PreTrainedModel):
    @classmethod
    def get_init_context(cls, is_quantized: bool, _is_ds_init_called: bool):
        if is_deepspeed_zero3_enabled():
            import turbo_alignment.modeling.parallel_states as mpu

            init_contexts = [no_init_weights()]
            # We cannot initialize the model on meta device with deepspeed when not quantized
            if not is_quantized and not _is_ds_init_called:
                logger.info("Detected DeepSpeed ZeRO-3: activating zero.init() for this model")
                init_contexts.extend(
                    [deepspeed.zero.Init(config_dict_or_path=deepspeed_config(), mpu=mpu), set_zero3_state()]
                )
            elif is_quantized:
                init_contexts.extend([init_empty_weights(), set_quantized_state()])
        else:
            init_contexts = [no_init_weights(), init_empty_weights()]

        return init_contexts

    @classmethod
    @restore_default_torch_dtype
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
        torch_dtype = kwargs.pop("torch_dtype", config.torch_dtype)
        if isinstance(torch_dtype, str):
            torch_dtype = getattr(torch, torch_dtype)

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

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "5GB",
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        save_peft_format: bool = True,
        **kwargs,
    ):
        """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        [`~PreTrainedModel.from_pretrained`] class method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to which to save. Will be created if it doesn't exist.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful when in distributed training like
                TPUs and need to call this function on all processes. In this case, set `is_main_process=True` only on
                the main process to avoid race conditions.
            state_dict (nested dictionary of `torch.Tensor`):
                The state dictionary of the model to save. Will default to `self.state_dict()`, but can be used to only
                save parts of the model or if special precautions need to be taken when recovering the state dictionary
                of a model (like when using model parallelism).
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful on distributed training like TPUs when one
                need to replace `torch.save` by another method.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            max_shard_size (`int` or `str`, *optional*, defaults to `"5GB"`):
                The maximum size for a checkpoint before being sharded. Checkpoints shard will then be each of size
                lower than this size. If expressed as a string, needs to be digits followed by a unit (like `"5MB"`).
                We default it to 5GB in order for models to be able to run easily on free-tier google colab instances
                without CPU OOM issues.

                <Tip warning={true}>

                If a single weight of the model is bigger than `max_shard_size`, it will be in its own checkpoint shard
                which will be bigger than `max_shard_size`.

                </Tip>

            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether to save the model using `safetensors` or the traditional PyTorch way (that uses `pickle`).
            variant (`str`, *optional*):
                If specified, weights are saved in the format pytorch_model.<variant>.bin.
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
                the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).
            save_peft_format (`bool`, *optional*, defaults to `True`):
                For backward compatibility with PEFT library, in case adapter weights are attached to the model, all
                keys of the state dict of adapters needs to be pre-pended with `base_model.model`. Advanced users can
                disable this behaviours by setting `save_peft_format` to `False`.
            kwargs (`Dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        use_auth_token = kwargs.pop("use_auth_token", None)
        ignore_metadata_errors = kwargs.pop("ignore_metadata_errors", False)

        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        if token is not None:
            kwargs["token"] = token

        _hf_peft_config_loaded = getattr(self, "_hf_peft_config_loaded", False)

        hf_quantizer = getattr(self, "hf_quantizer", None)
        quantization_serializable = (
            hf_quantizer is not None
            and isinstance(hf_quantizer, HfQuantizer)
            and hf_quantizer.is_serializable(safe_serialization=safe_serialization)
        )

        if hf_quantizer is not None and not _hf_peft_config_loaded and not quantization_serializable:
            raise ValueError(
                f"The model is quantized with {hf_quantizer.quantization_config.quant_method} and is not serializable - check out the warnings from"
                " the logger on the traceback to understand the reason why the quantized model is not serializable."
            )

        if "save_config" in kwargs:
            warnings.warn(
                "`save_config` is deprecated and will be removed in v5 of Transformers. Use `is_main_process` instead."
            )
            is_main_process = kwargs.pop("save_config")
        if safe_serialization and not is_safetensors_available():
            raise ImportError("`safe_serialization` requires the `safetensors library: `pip install safetensors`.")

        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)

        # Only save the model itself if we are using distributed training
        model_to_save = unwrap_model(self)

        # save the string version of dtype to the config, e.g. convert torch.float32 => "float32"
        # we currently don't use this setting automatically, but may start to use with v5
        dtype = get_parameter_dtype(model_to_save)
        model_to_save.config.torch_dtype = str(dtype).split(".")[1]

        # Attach architecture to the config
        model_to_save.config.architectures = [model_to_save.__class__.__name__.removesuffix('WithMPU')]  # PATCHED

        # Unset attn implementation so it can be set to another one when loading back
        model_to_save.config._attn_implementation_autoset = False

        # If we have a custom model, we copy the file defining it in the folder and set the attributes so it can be
        # loaded from the Hub.
        if self._auto_class is not None:
            custom_object_save(self, save_directory, config=self.config)

        # Save the config
        if is_main_process:
            if not _hf_peft_config_loaded:
                # If the model config has set attributes that should be in the generation config, move them there.
                misplaced_generation_parameters = model_to_save.config._get_non_default_generation_parameters()
                if self.can_generate() and len(misplaced_generation_parameters) > 0:
                    warnings.warn(
                        "Moving the following attributes in the config to the generation config: "
                        f"{misplaced_generation_parameters}. You are seeing this warning because you've set "
                        "generation parameters in the model config, as opposed to in the generation config.",
                        UserWarning,
                    )
                    for param_name, param_value in misplaced_generation_parameters.items():
                        setattr(model_to_save.generation_config, param_name, param_value)
                        setattr(model_to_save.config, param_name, None)

                model_to_save.config.save_pretrained(save_directory)
            if self.can_generate():
                model_to_save.generation_config.save_pretrained(save_directory)

            if _hf_peft_config_loaded:
                logger.info(
                    "Detected adapters on the model, saving the model in the PEFT format, only adapter weights will be saved."
                )
                state_dict = model_to_save.get_adapter_state_dict(state_dict=state_dict)

                if save_peft_format:
                    logger.info(
                        "To match the expected format of the PEFT library, all keys of the state dict of adapters will be pre-pended with `base_model.model`."
                    )
                    peft_state_dict = {}
                    for key, value in state_dict.items():
                        peft_state_dict[f"base_model.model.{key}"] = value
                    state_dict = peft_state_dict

                active_adapter = self.active_adapters()

                if len(active_adapter) > 1:
                    raise ValueError(
                        "Multiple active adapters detected, saving multiple active adapters is not supported yet. You can save adapters separately one by one "
                        "by iteratively calling `model.set_adapter(adapter_name)` then `model.save_pretrained(...)`"
                    )
                active_adapter = active_adapter[0]

                current_peft_config = self.peft_config[active_adapter]
                current_peft_config.save_pretrained(save_directory)

        # for offloaded modules
        module_map = {}

        # Save the model
        if state_dict is None:
            # if any model parameters are offloaded, make module map
            if (
                hasattr(self, "hf_device_map")
                and len(set(self.hf_device_map.values())) > 1
                and ("cpu" in self.hf_device_map.values() or "disk" in self.hf_device_map.values())
            ):
                warnings.warn(
                    "Attempting to save a model with offloaded modules. Ensure that unallocated cpu memory exceeds the `shard_size` (5GB default)"
                )
                for name, module in model_to_save.named_modules():
                    if name == "":
                        continue
                    module_state_dict = module.state_dict()

                    for key in module_state_dict:
                        module_map[name + f".{key}"] = module
            state_dict = model_to_save.state_dict()

        # Translate state_dict from smp to hf if saving with smp >= 1.10
        if IS_SAGEMAKER_MP_POST_1_10:
            for smp_to_hf, _ in smp.state.module_manager.translate_functions:
                state_dict = smp_to_hf(state_dict)

        # Handle the case where some state_dict keys shouldn't be saved
        if self._keys_to_ignore_on_save is not None:
            for ignore_key in self._keys_to_ignore_on_save:
                if ignore_key in state_dict.keys():
                    del state_dict[ignore_key]

        # Rename state_dict keys before saving to file. Do nothing unless overridden in a particular model.
        # (initially introduced with TimmWrapperModel to remove prefix and make checkpoints compatible with timm)
        state_dict = self._fix_state_dict_keys_on_save(state_dict)

        if safe_serialization:
            # Safetensors does not allow tensor aliasing.
            # We're going to remove aliases before saving
            ptrs = collections.defaultdict(list)
            for name, tensor in state_dict.items():
                # Sometimes in the state_dict we have non-tensor objects.
                # e.g. in bitsandbytes we have some `str` objects in the state_dict
                if isinstance(tensor, torch.Tensor):
                    ptrs[id_tensor_storage(tensor)].append(name)
                else:
                    # In the non-tensor case, fall back to the pointer of the object itself
                    ptrs[id(tensor)].append(name)

            # These are all the pointers of shared tensors
            if hasattr(self, "hf_device_map"):
                # if the model has offloaded parameters, we must check using find_tied_parameters()
                tied_params = find_tied_parameters(self)
                if tied_params:
                    tied_names = tied_params[0]
                    shared_ptrs = {
                        ptr: names for ptr, names in ptrs.items() if any(name in tied_names for name in names)
                    }
                else:
                    shared_ptrs = {}
            else:
                shared_ptrs = {ptr: names for ptr, names in ptrs.items() if len(names) > 1}

            # Recursively descend to find tied weight keys
            _tied_weights_keys = _get_tied_weight_keys(self)
            error_names = []
            to_delete_names = set()
            for names in shared_ptrs.values():
                # Removing the keys which are declared as known duplicates on
                # load. This allows to make sure the name which is kept is consistent.
                if _tied_weights_keys is not None:
                    found = 0
                    for name in sorted(names):
                        matches_pattern = any(re.search(pat, name) for pat in _tied_weights_keys)
                        if matches_pattern and name in state_dict:
                            found += 1
                            if found < len(names):
                                to_delete_names.add(name)
            # We are entering a place where the weights and the transformers configuration do NOT match.
            shared_names, disjoint_names = _find_disjoint(shared_ptrs.values(), state_dict)
            # Those are actually tensor sharing but disjoint from each other, we can safely clone them
            # Reloaded won't have the same property, but it shouldn't matter in any meaningful way.
            for name in disjoint_names:
                state_dict[name] = state_dict[name].clone()

            # When not all duplicates have been cleaned, still remove those keys, but put a clear warning.
            # If the link between tensors was done at runtime then `from_pretrained` will not get
            # the key back leading to random tensor. A proper warning will be shown
            # during reload (if applicable), but since the file is not necessarily compatible with
            # the config, better show a proper warning.
            shared_names, identical_names = _find_identical(shared_names, state_dict)
            # delete tensors that have identical storage
            for inames in identical_names:
                known = inames.intersection(to_delete_names)
                for name in known:
                    del state_dict[name]
                unknown = inames.difference(to_delete_names)
                if len(unknown) > 1:
                    error_names.append(unknown)

            if shared_names:
                error_names.append(set(shared_names))

            if len(error_names) > 0:
                raise RuntimeError(
                    f"The weights trying to be saved contained shared tensors {error_names} that are mismatching the transformers base configuration. Try saving using `safe_serialization=False` or remove this tensor sharing.",
                )

        # Shard the model if it is too big.
        if not _hf_peft_config_loaded:
            weights_name = SAFE_WEIGHTS_NAME if safe_serialization else WEIGHTS_NAME
            weights_name = _add_variant(weights_name, variant)
        else:
            weights_name = ADAPTER_SAFE_WEIGHTS_NAME if safe_serialization else ADAPTER_WEIGHTS_NAME

        filename_pattern = weights_name.replace(".bin", "{suffix}.bin").replace(".safetensors", "{suffix}.safetensors")
        state_dict_split = split_torch_state_dict_into_shards(
            state_dict, filename_pattern=filename_pattern, max_shard_size=max_shard_size
        )
        # Save index if sharded
        index = None
        if state_dict_split.is_sharded:
            index = {
                "metadata": state_dict_split.metadata,
                "weight_map": state_dict_split.tensor_to_filename,
            }

        # Clean the folder from a previous save
        for filename in os.listdir(save_directory):
            full_filename = os.path.join(save_directory, filename)
            # If we have a shard file that is not going to be replaced, we delete it, but only from the main process
            # in distributed settings to avoid race conditions.
            weights_no_suffix = weights_name.replace(".bin", "").replace(".safetensors", "")

            # make sure that file to be deleted matches format of sharded file, e.g. pytorch_model-00001-of-00005
            filename_no_suffix = filename.replace(".bin", "").replace(".safetensors", "")
            reg = re.compile(r"(.*?)-\d{5}-of-\d{5}")

            if (
                filename.startswith(weights_no_suffix)
                and os.path.isfile(full_filename)
                and filename not in state_dict_split.filename_to_tensors.keys()
                and is_main_process
                and reg.fullmatch(filename_no_suffix) is not None
            ):
                os.remove(full_filename)
        # Save the model
        filename_to_tensors = state_dict_split.filename_to_tensors.items()
        if module_map:
            filename_to_tensors = logging.tqdm(filename_to_tensors, desc="Saving checkpoint shards")
        for shard_file, tensors in filename_to_tensors:
            shard = {}
            for tensor in tensors:
                shard[tensor] = state_dict[tensor].contiguous()
                # delete reference, see https://github.com/huggingface/transformers/pull/34890
                del state_dict[tensor]

            # remake shard with onloaded parameters if necessary
            if module_map:
                if accelerate_version < version.parse("0.31"):
                    raise ImportError(
                        f"You need accelerate version to be greater or equal than 0.31 to save models with offloaded parameters. Detected version {accelerate_version}. "
                        f"Please upgrade accelerate with `pip install -U accelerate`"
                    )
                # init state_dict for this shard
                shard_state_dict = dict.fromkeys(shard, "")
                for module_name in shard:
                    # skip to collect this weight again
                    if shard_state_dict.get(module_name) != "":
                        continue
                    module = module_map[module_name]
                    # update state dict with onloaded parameters
                    shard_state_dict = get_state_dict_from_offload(module, module_name, shard_state_dict)

                # assign shard to be the completed state dict
                shard = shard_state_dict
                del shard_state_dict
                gc.collect()

            if safe_serialization:
                # At some point we will need to deal better with save_function (used for TPU and other distributed
                # joyfulness), but for now this enough.
                safe_save_file(shard, os.path.join(save_directory, shard_file), metadata={"format": "pt"})
            else:
                save_function(shard, os.path.join(save_directory, shard_file))

        del state_dict

        if index is None:
            path_to_weights = os.path.join(save_directory, weights_name)
            logger.info(f"Model weights saved in {path_to_weights}")
        else:
            save_index_file = SAFE_WEIGHTS_INDEX_NAME if safe_serialization else WEIGHTS_INDEX_NAME
            save_index_file = os.path.join(save_directory, _add_variant(save_index_file, variant))
            # Save the index as well
            with open(save_index_file, "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)
            logger.info(
                f"The model is bigger than the maximum size per checkpoint ({max_shard_size}) and is going to be "
                f"split in {len(state_dict_split.filename_to_tensors)} checkpoint shards. You can find where each parameters has been saved in the "
                f"index located at {save_index_file}."
            )

        if push_to_hub:
            # Eventually create an empty model card
            model_card = create_and_tag_model_card(
                repo_id, self.model_tags, token=token, ignore_metadata_errors=ignore_metadata_errors
            )

            # Update model card if needed:
            model_card.save(os.path.join(save_directory, "README.md"))

            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=token,
            )
