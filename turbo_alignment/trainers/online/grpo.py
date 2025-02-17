from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
import itertools
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.distributed
import torch.nn.functional as F
import torch.utils
import torch.utils.data
from datasets import Dataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
    TrainingArguments, 
    GenerationConfig,
)

from turbo_alignment.common.distributed import (
    get_global_mean,
    get_global_std,
    get_log_mean_std,
    _init_process_group
)
from turbo_alignment.generators import ChatGenerator, RMSamplingGenerator, vLLMChatGenerator, RayRMSamplingGenerator
from turbo_alignment.generators.base import ChatGeneratorBase
from turbo_alignment.settings.generators.chat import CustomChatGenerationSettings
from turbo_alignment.settings.generators.outputs.chat import ChatInferenceOutput
from turbo_alignment.common.data.io import read_jsonl
from turbo_alignment.settings.online import (
    CriticType,
    vLLMActorSettings,
    HFActorSettings,
    ActorType,
    RewardProcessorType,
)
from turbo_alignment.common.tf.loaders.model.model import disable_dropout_in_model
from turbo_alignment.settings.tf.generation import GeneratorTransformersSettings
from turbo_alignment.trainers.multigpu import MultiGPUCherryPicksTrainer
from turbo_alignment.trainers.online.reward_processor import GRPORewardProcessor, RewardProcessor
from turbo_alignment.trainers.online.ray.rayactor_group import RayGroup
from turbo_alignment.trainers.utils import prepare_model

import deepspeed
import ray
import socket
import os
from deepspeed.runtime.engine import DeepSpeedEngine
import gc
import time
import logging

def get_all_parameters(sub_module, recurse=False):
    return itertools.chain(sub_module.named_parameters(recurse=recurse), sub_module.ds_external_parameters())

def iter_params(module, recurse=False):
    return [param for _, param in get_all_parameters(module, recurse)]

def remove_hooks(model: "DeepSpeedEngine") -> None:
    """Removes the optimizer hooks from a DeepSpeed ZeRO-3 model."""
    if model.optimizer is not None and hasattr(model.optimizer, "parameter_offload"):
        optimizer_offload = model.optimizer.parameter_offload
    elif model.optimizer is not None:
        optimizer_offload = model.optimizer

    for param in iter_params(optimizer_offload.module, recurse=True):
        param.ds_active_sub_modules.clear()

    for hook in optimizer_offload.forward_hooks:
        hook.remove()
    for hook in optimizer_offload.backward_hooks:
        hook.remove()

    optimizer_offload.forward_hooks = []
    optimizer_offload.backward_hooks = []

def add_hooks(model: "DeepSpeedEngine") -> None:
    """Adds the optimizer hooks from a DeepSpeed ZeRO-3 model."""
    if model.optimizer is not None and hasattr(model.optimizer, "parameter_offload"):
        optimizer_offload = model.optimizer.parameter_offload
    elif model.optimizer is not None:
        optimizer_offload = model.optimizer
    optimizer_offload._register_hooks_recursively(optimizer_offload.module)

@contextmanager
def unwrap_model_for_generation(
    model,
    accelerator,
    is_peft_model: bool = False,
    gather_deepspeed3_params: bool = True,
):
    """Context manager to unwrap a model for generation.
    For ZeRO-3 models, we gather the weights once to speed up generation.
    """
    unwrapped_model = accelerator.unwrap_model(model)
    if is_peft_model:
        unwrapped_model.pretrained_model.disable_adapter()
    if accelerator.state.deepspeed_plugin is not None and accelerator.state.deepspeed_plugin.zero_stage == 3:
        if not gather_deepspeed3_params:
            yield accelerator.unwrap_model(model)
        else:
            with deepspeed.zero.GatheredParameters(model.parameters()):
                remove_hooks(model)
                yield accelerator.unwrap_model(model)
                add_hooks(model)
    else:
        yield unwrapped_model

def sum_all_parameter_values(model):
    if isinstance(model, deepspeed.DeepSpeedEngine):
        model = model.module
    total_sum = sum(p.sum().item() for p in model.parameters())
    return total_sum


class TimeProfiler:
    def __init__(self):
        self.broadcast_time = []
        self.completions_time = []
        self.reward_model_time = []
        self.reward_processing_time = []
        self.baseline_reward_time = []
        self.reference_model_time = []
        self.reference_forward_time = []
        self.policy_model_time = []
        self.policy_forward_time = []
        self.padding_time = []
        self.metrics_time = []
        self.total_time = []

    def print(self):
        import numpy as np
        
        additional_time = []
        
        for length in range(len(self.total_time)):
            other_time = 2 * self.total_time[length] - sum([self.__getattribute__(key)[length] for key in self.__dict__.keys()])
            additional_time.append(other_time)

        def mean_std_format(values):
            mean = np.mean(values)
            std = np.std(values)
            return f"{mean:.2f} (±{std:.2f})"

        print(f"Broadcast Time: {mean_std_format(self.broadcast_time)}", flush=True)
        print(f"Completions Time: {mean_std_format(self.completions_time)}", flush=True)
        print(f"Reward Model Time: {mean_std_format(self.reward_model_time)}", flush=True)
        print(f"Reward Processing Time: {mean_std_format(self.reward_processing_time)}", flush=True)
        print(f"Reward Baseline Time: {mean_std_format(self.baseline_reward_time)}", flush=True)
        print(f"Reference Logprobs Time: {mean_std_format(self.reference_model_time)}", flush=True)
        print(f"Reference Forward Time: {mean_std_format(self.reference_forward_time)}", flush=True)
        print(f"Policy Logprobs Time: {mean_std_format(self.policy_model_time)}", flush=True)
        print(f"Policy Forward Time: {mean_std_format(self.policy_forward_time)}", flush=True)
        print(f"Total Time: {mean_std_format(self.total_time)}", flush=True)
        print(f"Padding Time: {mean_std_format(self.padding_time)}", flush=True)
        print(f"Metrics Time: {mean_std_format(self.metrics_time)}", flush=True)
        print(f"Additional Time: {mean_std_format(additional_time)}", flush=True)


# FIXME
@dataclass
class GRPOTrainingArguments(TrainingArguments):
    num_nodes: int = 2
    reward_model_replicas: int = 1
    reference_model_replicas: int = 1
    max_new_tokens: int = 1024
    stop_token: str = '<eos>'

    penalty_reward_value: float = 0.1
    clip_rewards_min: float = 0.1
    clip_rewards_max: float = 1.0
    kl_coef: float = 0.05

    num_generations: int = 3
    num_samples_for_reward_stats: int = 0

    non_eos_penalty: bool = True
    temperature: float | None = None
    whiten_rewards: bool = False

    actor_type: ActorType = ActorType.DISTRIBUTED_VLLM
    critic_type: CriticType = CriticType.RAY_TRANSFORMERS
    actor_settings: vLLMActorSettings | HFActorSettings = vLLMActorSettings

class GRPOTrainer(MultiGPUCherryPicksTrainer):
    def __init__(
        self,
        vllm_engines: list,
        args: GRPOTrainingArguments,
        processing_class: PreTrainedTokenizerBase,
        policy: PreTrainedModel | torch.nn.Module,
        reward_model: PreTrainedModel | torch.nn.Module,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        ref_model: PreTrainedModel | torch.nn.Module | None = None,
        callbacks: list[TrainerCallback] | None = None,
        **kwargs,
    ) -> None:

        super().__init__(
            model=policy,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            **kwargs,
        )
        
        self.time_profiler = TimeProfiler()
        self.vllm_engines = vllm_engines

        if self.vllm_engines is not None and torch.distributed.get_rank() == 0:
            master_address = ray._private.services.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]
            
            # TODO_RLOO assert tp_size same for all engines
            world_size = args.actor_settings["vllm_num_engines"] * args.actor_settings["vllm_tensor_parallel_size"] + 1

            refs = [
                engine.init_weight_update_group.remote(
                    master_address,
                    master_port,
                    i * args.actor_settings["vllm_tensor_parallel_size"] + 1,
                    world_size,
                )
                for i, engine in enumerate(self.vllm_engines)
            ]

            from turbo_alignment.trainers.online.ray.vllm_worker_wrap import stateless_init_process_group
            # https://github.com/vllm-project/vllm/issues/11399
            # https://github.com/vllm-project/vllm/pull/12084
            # https://github.com/vllm-project/vllm/issues/5723
            self.model_update_group = stateless_init_process_group(
                master_address=master_address,
                master_port=master_port,
                world_size=world_size,
                rank=0,
                device=torch.device(f"cuda:0")
            )

            ray.get(refs)

        torch.distributed.barrier()

        self.ref_model = ref_model

        # TODO: TODO_RLOO watch later
        if self.ref_model is not None and not isinstance(self.ref_model, RayGroup):
            self.ref_model = prepare_model(self.ref_model, self.accelerator, self.is_deepspeed_enabled)
            disable_dropout_in_model(self.ref_model)

        elif isinstance(self.ref_model, RayGroup):
            refs = ray.get(
                self.ref_model.prepare_reference_model(self.accelerator, self.is_deepspeed_enabled)
            )
        
        disable_dropout_in_model(self.model)

        self.reward_model = reward_model

        self._stored_metrics: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        self.kl_coef = args.kl_coef

        self.gen_id = 1

        # TODO: Take gradient accumulation into account
        # In this way, the effective number of gradient
        # updates taken into account stays the same
        # effective_num_previous_samples = 1 / (1 - self.args.mean_baseline_coef)
        # self.args.mean_baseline_coef = 1 - 1 / (effective_num_previous_samples * self.args.gradient_accumulation_steps)

        self.stop_generation_token_id = processing_class.encode(args.stop_token, add_special_tokens=False)
        assert len(self.stop_generation_token_id) == 1, self.stop_generation_token_id
        
        #TODO separate stop_strings and eos_token
        
        self.generator_transformers_settings = GeneratorTransformersSettings(
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            num_return_sequences=args.num_generations,
            num_beams=args.num_generations,
            do_sample=True,
            stop_strings=args.stop_token
        )
        self.generator_custom_settings = CustomChatGenerationSettings(
            batch=self.args.per_device_train_batch_size,
            remove_prompt=False,
            only_answer_logits=True,
            return_logits=True,
        )
        self.reward_processor = GRPORewardProcessor(
            num_generations=args.num_generations,
        )

        self.critic_generator = self._get_rm_generator(self.reward_model)
        
        print("Generations Params:\n" + "\n".join([f"{attr}: {getattr(self.generator_transformers_settings, attr, None)}" for attr, _ in self.generator_transformers_settings.__annotations__.items()]))

        start = time.time()
        
        # if num_samples_for_reward_stats == 0 then no normalization is done FIXME

        self.norm_reward_mean, self.norm_reward_std = self.reward_stats(
            model=self.model, dataloader=self.get_train_dataloader()
        )
        logging.info(f'Statictis calculation elapsed time:{time.time() - start}')
    
    def _broadcast_to_vllm(self, model: DeepSpeedEngine):
        # avoid OOM
        # torch.cuda.empty_cache()
        model = model.module
        for name, param in model.named_parameters():
            # FIXME i dont know will it work with zero-3
            if torch.distributed.get_rank() == 0:
                refs = [
                    engine.collective_rpc.remote(
                        "update_weight",
                        args=(name, param.dtype, param.shape)
                    )
                    for engine in self.vllm_engines
                ]
                self.model_update_group.broadcast(
                    param, 
                    src=0,
                    stream=torch.cuda.current_stream()
                )
                ray.get(refs)

    
    # FIXME: some base class instead of RMSamplingGenerator (join with ChatGeneratorBase?)
    def _get_rm_generator(self, reward_model: torch.nn.Module | PreTrainedModel) -> RMSamplingGenerator:
        match self.args.critic_type:
            case CriticType.LOCAL_TRANSFORMERS:
                generator = RMSamplingGenerator(
                    model=reward_model,
                    tokenizer=self.processing_class,
                    accelerator=self.accelerator,
                )
            case CriticType.RAY_TRANSFORMERS:
                generator = RayRMSamplingGenerator(
                    model=reward_model,
                    tokenizer=self.processing_class,
                    model_replicas=self.args.reward_model_replicas,
                )#TODO this type of critic is created for Reward models with CausalLM head and utilize vllm engines
            case CriticType.DISTRIBUTED_VLLM:
                generator = ...
            case _:
                raise ValueError(f'Critic {self.args.critic_type} is not supported')
        return generator

    #TODO_RLOO why every time generates new object? Since the model weights are changing, recreate new object and 
    def _get_chat_generator(self, model: torch.nn.Module | PreTrainedModel = None) -> ChatGeneratorBase:
        match self.args.actor_type:
            case ActorType.LOCAL_TRANSFORMERS:
                generator = ChatGenerator(
                    model=model,
                    tokenizer=self.processing_class,
                    transformers_settings=self.generator_transformers_settings,
                    custom_generation_settings=self.generator_custom_settings,
                    accelerator=self.accelerator,
                )
            case ActorType.DISTRIBUTED_VLLM:
                generator = vLLMChatGenerator(
                    vllm_engines=self.vllm_engines,
                    tokenizer=self.processing_class,
                    transformers_settings=self.generator_transformers_settings,
                    custom_generation_settings=self.generator_custom_settings,
                )
            case _:
                raise ValueError(f'LLM Actor {self.args.actor_type} is not supported')
        return generator

    def get_answers_and_rewards(
        self,
        model: torch.nn.Module | PreTrainedModel | DeepSpeedEngine ,
        inputs: dict[str, torch.Tensor],
        do_broadcast=True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # if torch.distributed.get_rank() == 0:
        #     print(f'Input shape: {inputs["input_ids"].shape}', flush=True)
        #     print(f'Input ids example at index [0]: {inputs["input_ids"][0, :]}')
        #     print(f'Input Example at index [0]: {self.processing_class.batch_decode(inputs["input_ids"][0, :].unsqueeze(0))}')

        if do_broadcast:
            # TODO: move to generator
            torch.distributed.barrier()
            if torch.distributed.get_rank() == 0:
                start = time.time()

            self._broadcast_to_vllm(model)
            torch.distributed.barrier()

            if torch.distributed.get_rank() == 0:
                end = time.time()
                self.time_profiler.broadcast_time.append(end - start)


        generator = self._get_chat_generator()

        generations: list[ChatInferenceOutput] = generator.generate_from_batch_records(
            dataset_name='online', records=inputs, original_records=True, time_profiler = self.time_profiler
        )
        

        response_ids = [torch.cat([g.input_token_ids, ans.answer_token_ids], dim=1) for g in generations for ans in g.answers]
        response_attention_mask = [torch.cat([g.input_attention_mask, ans.answer_attention_mask], dim=1) for g in generations for ans in g.answers]

        # if torch.distributed.get_rank() == 0:
        #     print('response_ids.device, ', response_ids.device)
        #     print('response_attention_mask.device, ', response_attention_mask.device)
        # torch.distributed.barrier()

        # if torch.distributed.get_rank() == 0:
        #     print(f'Prompt with completion at index [0] shape: {response_ids[0].shape}', flush=True)
        #     print(f'Prompt with completion decoded: {self.tokenizer.batch_decode(response_ids[0])}', flush=True)

        max_length = max([response_id.size(1) for response_id in response_ids])

        def pad_sequences(sequences, max_length, pad_value=0):
            padded_sequences = []
            for seq in sequences:
                padding_needed = max_length - seq.size(1)
                padded_seq = torch.cat([seq.squeeze(0), torch.full((padding_needed,), pad_value, dtype=seq.dtype, device=seq.device)])
                padded_sequences.append(padded_seq)
            return torch.stack(padded_sequences)
        
        if torch.distributed.get_rank() == 0:
            start = time.time()
            
        # if torch.distributed.get_rank() == 0:
        #     print('response_ids.device, ', response_ids.device)
        # torch.distributed.barrier()
        response_ids = pad_sequences(response_ids, max_length, pad_value=self.tokenizer.pad_token_id)
        response_attention_mask = pad_sequences(response_attention_mask, max_length, pad_value=0)

        response_tokens_mask = torch.zeros(response_ids.shape, dtype=torch.bfloat16, device=response_ids.device) # why bf16 FIXME

        input_lengths = [g.input_token_ids.size(1) for g in generations for _ in g.answers]

        for i, input_len in enumerate(input_lengths):
            response_tokens_mask[i, input_len:] = 1.0 #FIXME 

        position_ids = (response_attention_mask.cumsum(-1) - 1).clamp(min=0)
        position_ids.masked_fill_(response_attention_mask.to(torch.bool) == 0, 0).cuda()

        if torch.distributed.get_rank() == 0:
            end = time.time()
            self.time_profiler.padding_time.append(end - start)

        rm_inputs = {
            'input_ids': response_ids,
            'attention_mask': response_attention_mask,
            'position_ids': position_ids,
        }
        
        if torch.distributed.get_rank() == 0:
            start = time.time()

        rewards = self.critic_generator.generate_from_batch_records(rm_inputs)

        if torch.distributed.get_rank() == 0:
            end = time.time()
            self.time_profiler.reward_model_time.append(end - start)
        
        return response_ids, response_attention_mask, response_tokens_mask, position_ids, rewards

    def get_batch_loss_metrics(
        self,
        model: torch.nn.Module | PreTrainedModel,
        inputs: dict[str, torch.Tensor],
        train_eval: Literal['train', 'eval'] = 'train',
    ):
        
        with torch.no_grad():
            (
                query_response,
                attention_mask,
                response_tokens_mask,
                position_ids,
                rewards,
            ) = self.get_answers_and_rewards(
                model=model,
                inputs=inputs,
                do_broadcast=True if train_eval == 'train' else False
            )
            
            if isinstance(self.ref_model, RayGroup):
                ref_logits = ray.get(
                    self.ref_model.reference_forward(
                        {
                            'input_ids': query_response,
                            'attention_mask': attention_mask,
                            'position_ids': position_ids,
                            'use_cache': False
                        },
                        index=torch.distributed.get_rank() % self.args.reference_model_replicas,
                    )
                )
            else:
                if torch.distributed.get_rank() == 0:
                    start = time.time()

                ref_logits = self.ref_model(
                    input_ids=query_response,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                ).logits[:, :-1]

                if torch.distributed.get_rank() == 0:
                    end = time.time()
                    self.time_profiler.reference_forward_time.append(end - start)

            if torch.distributed.get_rank() == 0:
                start = time.time()

            ref_logprobs = self.get_logprobs(
                logits=ref_logits,
                input_ids=query_response,
                loss_mask=response_tokens_mask,
                do_sum=False,
            )

            if torch.distributed.get_rank() == 0:
                end = time.time()
                self.time_profiler.reference_model_time.append(end - start)
            
            if torch.distributed.get_rank() == 0:
                start = time.time()

            rewards, valid_mask, rewards_metrics = self.process_rewards(rewards=rewards, query_response=query_response)

            if torch.distributed.get_rank() == 0:
                end = time.time()
                self.time_profiler.reward_processing_time.append(end - start)

        if torch.distributed.get_rank() == 0:
            start = time.time()

        logits = model(
            input_ids=query_response,
            attention_mask=attention_mask,
            position_ids=position_ids,
        ).logits[:, :-1]

        if torch.distributed.get_rank() == 0:
            end = time.time()
            self.time_profiler.policy_forward_time.append(end - start)
        
        # del inputs
        # gc.collect()
        # torch.cuda.empty_cache() #NEED?
        
        if torch.distributed.get_rank() == 0:
            start = time.time()

        logprobs = self.get_logprobs(
            logits=logits,
            input_ids=query_response,
            loss_mask=response_tokens_mask,
            do_sum=False,
        )

        print('logprobs.shape', logprobs.shape)
        print('ref_logprobs.shape', ref_logprobs.shape)

        if torch.distributed.get_rank() == 0:
            end = time.time()
            self.time_profiler.policy_model_time.append(end - start)

        with torch.no_grad():
            kl_term = (torch.exp(ref_logprobs - logprobs) - (ref_logprobs - logprobs) - 1)

            print('kl_term.shape', kl_term.shape)

            if torch.distributed.get_rank() == 0:
                start = time.time()

            advantages, advantages_metrics = self.reward_processor.baseline_rewards(rewards=rewards) # its advantage

            if torch.distributed.get_rank() == 0:
                end = time.time()
                self.time_profiler.baseline_reward_time.append(end - start)

        print('advantages.shape', advantages.shape)

        per_token_loss = torch.exp(logprobs - logprobs.detach()) * advantages.unsqueeze(1) # https://github.com/huggingface/trl/pull/2565#issuecomment-2595837761
        per_token_loss = -(per_token_loss - self.kl_coef * kl_term)
        loss = per_token_loss.mean(dim=1)

        print('per_token_loss.shape', per_token_loss.shape)
        print('loss.shape', loss.shape)

        assert len(loss.shape) == 1, loss.shape

        if torch.distributed.get_rank() == 0:
            start = time.time()

        with torch.no_grad():
            metrics = {}
            for tensor, name in zip(
                [
                    rewards,
                    advantages,
                    loss.detach(),
                    kl_term.mean().detach(),
                    logprobs.detach(),
                    ref_logprobs,
                    1 - valid_mask.float(),
                ],
                [
                    'rewards',
                    'advantages',
                    'loss',
                    'kl_term',
                    'logprobs',
                    'ref_logprobs',
                    'invalid_rewards',
                ],
            ):
                metrics = {
                    **metrics,
                    **get_log_mean_std(tensor.cuda(), name, train_eval),
                }

            for k, v in rewards_metrics.items():
                metrics[f'{train_eval}/{k}'] = v
            for k, v in advantages_metrics.items():
                metrics[f'{train_eval}/{k}'] = v
            
            for k, v in metrics.items():
                if isinstance(v, torch.Tensor):
                    metrics[k] = metrics[k]

        if torch.distributed.get_rank() == 0:
            end = time.time()
            self.time_profiler.metrics_time.append(end - start)
        
        return loss.mean(), metrics
    
    def print_readable_stats(self):
        if torch.distributed.get_rank() == 0:
            print(f"Allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
            print(f"Reserved: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")

    def compute_loss(self, model, inputs, return_outputs: bool = False, num_items_in_batch=None):        
        if torch.distributed.get_rank() == 0:
            start = time.time()

        loss, metrics = self.get_batch_loss_metrics(model, inputs, 'train')

        self.store_metrics(metrics=metrics, train_eval='train')
        
        if torch.distributed.get_rank() == 0:
            end = time.time()
            self.time_profiler.total_time.append(end - start)
            # logging.info(f'Total elapsed time:{end - start}')
            self.time_profiler.print()
        
        # gc.collect()
        # torch.cuda.empty_cache() #NEED?

        return (loss, metrics) if return_outputs else loss

    def prediction_step(
        self,
        model: Union[PreTrainedModel, torch.nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        with torch.no_grad():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, 'eval')

        self.store_metrics(metrics, train_eval='eval')

        return loss.detach(), None, None

    @torch.no_grad()
    def reward_stats(
        self, model: torch.nn.Module | PreTrainedModel, dataloader: torch.utils.data.DataLoader
    ) -> tuple[float, float]:
        if self.args.num_samples_for_reward_stats == 0:
            norm_reward_mean = 0.0
            norm_reward_std = 1.0
        else:
            all_rewards = []
            samples_processed = 0
            for batch in dataloader:
                _, _, _, _, rewards = self.get_answers_and_rewards(
                    model=model,
                    inputs=batch,
                    do_broadcast=False
                )
                rewards, _ = self.reward_processor.postprocess_rewards(rewards=rewards)

                all_rewards.append(rewards)
                samples_processed += len(batch['input_ids']) * self.accelerator.num_processes

                if samples_processed > self.args.num_samples_for_reward_stats:
                    break

            all_rewards = torch.cat(all_rewards, 0)

            norm_reward_metrics = get_log_mean_std(all_rewards, 'norm_reward', train_eval='train', use_global=True)

            norm_reward_mean = norm_reward_metrics['train/norm_reward_mean']
            norm_reward_std = norm_reward_metrics['train/norm_reward_std']

        return norm_reward_mean, norm_reward_std

    def get_logprobs(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        log_softmax: bool = True,
        do_sum: bool = True,
    ):
        logits /= self.args.temperature

        all_logprob = F.log_softmax(logits, dim=-1) if log_softmax else F.softmax(logits, dim=-1)

        logprob = torch.gather(all_logprob, 2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        logprob[~loss_mask[:, 1:].to(torch.bool)] = 0

        if do_sum:
            return logprob.sum(-1)
            
        return logprob

    # FIXME
    def fill_nonvalid_rewards(self, rewards, query_response) -> Tuple[torch.Tensor, torch.Tensor]:
        # if self.args.non_eos_penalty:
        #     invalid_mask = query_response[:, -1] != self.stop_generation_token_id[0]
        #     rewards[invalid_mask] = self.args.penalty_reward_value

        #     return rewards, ~invalid_mask

        return rewards, torch.ones_like(rewards).to(torch.bool)

    def process_rewards(self, rewards, query_response) -> tuple[torch.Tensor, torch.Tensor, Any]: #bottleneck
        """
        Second, this token should be the <eot_id>.
        Otherwise, return a fixed reward of -1.
        """

        rewards, reward_metrics = self.reward_processor.postprocess_rewards(rewards=rewards) #FIXME 4 secods?
        rewards = rewards.squeeze(-1)
        reward_metrics['normalizing_reward_mean'] = self.norm_reward_mean
        reward_metrics['normalizing_reward_std'] = self.norm_reward_std
        rewards = (rewards - self.norm_reward_mean) / self.norm_reward_std
        # boundness is required: https://arxiv.org/pdf/2406.01462
        rewards = torch.clamp(rewards, self.args.clip_rewards_min, self.args.clip_rewards_max)

        rewards, valid_mask = self.fill_nonvalid_rewards(rewards=rewards, query_response=query_response)

        return rewards, valid_mask, reward_metrics

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal['train', 'eval'] = 'train') -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def log(self, logs: Dict[str, float], start_time: float | None = None) -> None:
        main_keys = list(self._stored_metrics.keys())
        for main_key in main_keys:
            for key, metrics in self._stored_metrics[main_key].items():
                if not isinstance(metrics, (list, tuple)):
                    metrics = [metrics] 


                float_values = []
                tensor_values = []
                for m in metrics:
                    if isinstance(m, float):
                        float_values.append(m)
                    elif isinstance(m, (int, bool)):
                        float_values.append(float(m))
                    elif isinstance(m, torch.Tensor):
                        tensor_values.append(m)
                    else:
                        logging.info(f'skipped metric {m} in logging.')

                if tensor_values:
                    gpu_tensors = [t.cuda() for t in tensor_values]
                    stacked = torch.stack(gpu_tensors)
                    gpu_mean = stacked.mean()
                    metric_val = gpu_mean.detach().cpu().item()
                    logs[key] = metric_val
                elif float_values:
                    # If we have no tensors but have floats, just log the last float.
                    logs[key] = float_values[-1]
                else:
                    logging.info(f'skipped key {key} in logging.')
                    logs[key] = float('nan')

            if main_key == 'train':
                logs['train/global_step'] = int(self.state.global_step)
            else:
                logs['eval/global_step'] = int(self.state.global_step // self.args.eval_steps)

            del self._stored_metrics[main_key]

        return super().log(logs)  # pylint: disable=no-member
