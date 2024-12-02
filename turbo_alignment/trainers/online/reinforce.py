from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils
import torch.utils.data
from datasets import Dataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
    TrainingArguments, GenerationConfig,
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
from turbo_alignment.settings.online import (
    CriticType,
    vLLMActorSettings,
    HFActorSettings,
    ActorType,
    RewardProcessorType,
)

from turbo_alignment.settings.tf.generation import GeneratorTransformersSettings
from turbo_alignment.trainers.multigpu import MultiGPUCherryPicksTrainer
from turbo_alignment.trainers.online.reward_processor import RewardProcessor
from turbo_alignment.trainers.utils import prepare_model

import deepspeed
import ray
import socket
import os
from deepspeed.runtime.engine import DeepSpeedEngine
import gc


def disable_dropout_in_model(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0

# FIXME
@dataclass
class REINFORCETrainingArguments(TrainingArguments):
    max_new_tokens: int = 1024
    stop_token: str = '<eos>'

    penalty_reward_value: float = 0.1
    clip_rewards_min: float = 0.1
    clip_rewards_max: float = 1.0
    kl_coef: float = 0.05
    mean_baseline_coef: float = 0.1

    num_generations: int = 3
    num_samples_for_reward_stats: int = 0

    non_eos_penalty: bool = True
    temperature: float | None = None
    whiten_rewards: bool = False

    actor_type: ActorType = ActorType.DISTRIBUTED_VLLM
    critic_type: CriticType = CriticType.RAY_TRANSFORMERS
    actor_settings: vLLMActorSettings | HFActorSettings = vLLMActorSettings

    reward_processor_type: RewardProcessorType = RewardProcessorType.RLOO

class REINFORCETrainer(MultiGPUCherryPicksTrainer):
    def __init__(
        self,
        vllm_engines: list,
        args: REINFORCETrainingArguments,
        tokenizer: PreTrainedTokenizerBase,
        policy: PreTrainedModel | torch.nn.Module,
        reward_model: PreTrainedModel | torch.nn.Module,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        ref_model: PreTrainedModel | torch.nn.Module | None = None,
        callbacks: list[TrainerCallback] | None = None,
        **kwargs,
    ) -> None:
        import time
        import logging
        
        start = time.time()

        super().__init__(
            model=policy,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            callbacks=callbacks,
            **kwargs,
        )
        logging.info(f'super().__init__ elapsed time:{time.time() - start}')
        
        self.vllm_engines = vllm_engines
        start = time.time()
        if self.vllm_engines is not None and torch.distributed.get_rank() == 0:
            master_address = ray._private.services.get_node_ip_address()
            print(f'TRAINER DEBUG: {master_address=}', flush=True)
            with socket.socket() as sock:
                sock.bind(("", 0))
                master_port = sock.getsockname()[1]
            
            # TODO_RLOO assert tp_size same for all engines
            world_size = args.actor_settings["vllm_num_engines"] * args.actor_settings["vllm_tensor_parallel_size"] + 1
            
            #TODO turn on nccl

            backend = "nccl"
            # https://github.com/OpenRLHF/OpenRLHF/issues/313
            import vllm

            if vllm.__version__ > "0.4.2" and os.getenv("NCCL_P2P_DISABLE", "0") == "0":
                backend = "gloo"
                print(
                    "Warning: using --vllm_sync_backend=gloo for vLLM version > 0.4.2 (or export NCCL_P2P_DISABLE=1)"
                )
            else:
                print("Using NCCL backend")

            refs = [
                engine.init_process_group.remote(
                    master_address,
                    master_port,
                    i * args.actor_settings["vllm_tensor_parallel_size"] + 1,
                    world_size,
                    "rloo",
                    backend=backend,
                )
                for i, engine in enumerate(self.vllm_engines)
            ]
            self._model_update_group = _init_process_group(
                backend=backend,
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=world_size,
                rank=0,
                group_name="rloo",
            )

            ray.get(refs)

        torch.distributed.barrier()
        logging.info(f'distributed vllm engine __init__ elapsed time:{time.time() - start}')

        self.ref_model = ref_model
        # TODO: delete later
        # ray.get(self.ref_model.async_eval())

        # TODO: TODO_RLOO watch later
        # if self.ref_model is not None:
        #     self.ref_model = prepare_model(self.ref_model, self.accelerator, self.is_deepspeed_enabled)
        disable_dropout_in_model(self.model)

        self.reward_model = reward_model
        # TODO: delete later
        # ray.get(self.ref_model.async_eval())

        self._stored_metrics: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        self.kl_coef = args.kl_coef

        self.gen_id = 1

        # Take gradient accumulation into account
        # In this way, the effective number of gradient
        # updates taken into account stays the same
        effective_num_previous_samples = 1 / (1 - self.args.mean_baseline_coef)
        self.args.mean_baseline_coef = 1 - 1 / (effective_num_previous_samples * self.args.gradient_accumulation_steps)

        self.stop_generation_token_id = tokenizer.encode(args.stop_token, add_special_tokens=False)
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
        self.reward_processor = RewardProcessor.by_name(args.reward_processor_type)(
            mean_baseline_coef=args.mean_baseline_coef,
            num_generations=args.num_generations,
        )
        start = time.time()
        self.print_readable_stats()
        self.norm_reward_mean, self.norm_reward_std = self.reward_stats(
            model=self.model, dataloader=self.get_train_dataloader()
        )
        logging.info(f'statictis in __init__ elapsed time:{time.time() - start}')
        self.print_readable_stats()

    
    def _broadcast_to_vllm(self, model: DeepSpeedEngine):
        # avoid OOM
        torch.cuda.empty_cache()
        model = model.module
        count, num_params = 0, len(list(model.named_parameters()))
        for name, param in model.named_parameters():
            count += 1  # empty_cache at last param

            # Fire all vllm engines for broadcast
            if torch.distributed.get_rank() == 0:
                shape = param.shape if self.accelerator.deepspeed_plugin.zero_stage != 3 else param.ds_shape
                refs = [
                    engine.update_weight.remote(name, dtype=param.dtype, shape=shape, empty_cache=count == num_params)
                    for engine in self.vllm_engines
                ]

            # For ZeRO-3, allgather sharded parameter and broadcast to all vllm engines by rank 0
            with deepspeed.zero.GatheredParameters([param], enabled=self.accelerator.deepspeed_plugin.zero_stage == 3):
                if torch.distributed.get_rank() == 0:
                    torch.distributed.broadcast(param.data, 0, group=self._model_update_group)
                    ray.get(refs)
    
    # FIXME: some base class instead of RMSamplingGenerator (join with ChatGeneratorBase?)
    def _get_rm_generator(self, reward_model: torch.nn.Module | PreTrainedModel) -> RMSamplingGenerator:
        match self.args.critic_type:
            case CriticType.LOCAL_TRANSFORMERS:
                generator = RMSamplingGenerator(
                    model=reward_model,
                    tokenizer=self.tokenizer,
                    accelerator=self.accelerator,
                )
            case CriticType.RAY_TRANSFORMERS:
                generator = RayRMSamplingGenerator(
                    model=reward_model,
                    tokenizer=self.tokenizer,
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
                    tokenizer=self.tokenizer,
                    transformers_settings=self.generator_transformers_settings,
                    custom_generation_settings=self.generator_custom_settings,
                    accelerator=self.accelerator,
                )
            case ActorType.DISTRIBUTED_VLLM:
                generator = vLLMChatGenerator(
                    vllm_engines=self.vllm_engines,
                    tokenizer=self.tokenizer,
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
        import time
        import logging

        if torch.distributed.get_rank() == 0:
            print(f'Input shape: {inputs["input_ids"].shape}', flush=True)
            print(f'Input Example at index [0]: {self.tokenizer.batch_decode(inputs["input_ids"][0, :].unsqueeze(0))}')

        if do_broadcast:
            # TODO: move to generator
            torch.distributed.barrier()
            start = time.time()
            self._broadcast_to_vllm(model)
            torch.distributed.barrier()
            logging.info(f'broadcast elapsed time:{time.time() - start}')

        start = time.time()
        generator = self._get_chat_generator()

        generations: list[ChatInferenceOutput] = generator.generate_from_batch_records(
            dataset_name='online', records=inputs, original_records=True
        )
        logging.info(f'generations elapsed time:{time.time() - start}')
        
        for g in generations:
            for ans in g.answers:
                print(f'{g.input_token_ids.device=}, {ans.answer_token_ids.device=}', flush=True)
                print(f'{g.input_attention_mask.device=}, {ans.answer_attention_mask.device=}', flush=True)
                break
            break
        response_ids = [torch.cat([g.input_token_ids, ans.answer_token_ids], dim=1) for g in generations for ans in g.answers]
        response_attention_mask = [torch.cat([g.input_attention_mask, ans.answer_attention_mask], dim=1) for g in generations for ans in g.answers]
        
        import random
        ind = random.randint(0, len(response_ids) - 1)
        assert response_ids[ind].shape == response_attention_mask[ind].shape

        if torch.distributed.get_rank() == 0:
            print(f'Prompt with completion at index [0] shape: {response_ids[0].shape}', flush=True)
            print(f'Prompt with completion decoded: {self.tokenizer.batch_decode(response_ids[0])}', flush=True)

        # Padding
        max_length = max([response_id.size(1) for response_id in response_ids])
        logging.info(f'{max_length=}')

        def pad_sequences(sequences, max_length, pad_value=0):
            padded_sequences = []
            for seq in sequences:
                padding_needed = max_length - seq.size(1)
                padded_seq = torch.cat([seq.squeeze(0), torch.full((padding_needed,), pad_value, dtype=seq.dtype)])
                padded_sequences.append(padded_seq)
            return torch.stack(padded_sequences)

        response_ids = pad_sequences(response_ids, max_length, pad_value=self.tokenizer.pad_token_id)
        response_attention_mask = pad_sequences(response_attention_mask, max_length, pad_value=0)

        response_tokens_mask = torch.zeros(response_ids.shape, dtype=torch.bfloat16)
        response_tokens_mask[:, generations[0].input_token_ids.shape[0] :] = 1.0

        position_ids = (response_attention_mask.cumsum(-1) - 1).clamp(min=0)
        position_ids.masked_fill_(response_attention_mask.to(torch.bool) == 0, 0)

        rm_inputs = {
            'input_ids': response_ids,
            'attention_mask': response_attention_mask,
            'position_ids': position_ids,
        }
        
        start = time.time()
        critic_generator = self._get_rm_generator(self.reward_model)
        rewards = critic_generator.generate_from_batch_records(rm_inputs)

        logging.info(f'rewards elapsed time:{time.time() - start}')
        return response_ids, response_attention_mask, response_tokens_mask, position_ids, rewards

    def get_batch_loss_metrics(
        self,
        model: torch.nn.Module | PreTrainedModel,
        inputs: dict[str, torch.Tensor],
        train_eval: Literal['train', 'eval'] = 'train',
    ):
        import logging
        import time
        
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
            start = time.time()
            ref_logprobs = self.get_logprobs(
                model=self.ref_model,
                input_ids=query_response,
                attention_mask=attention_mask,
                position_ids=position_ids,
                loss_mask=response_tokens_mask,
            )
            logging.info(f'reference elapsed time:{time.time() - start}')
            rewards, valid_mask, rewards_metrics = self.process_rewards(rewards=rewards, query_response=query_response)
        
        start = time.time()
        del inputs
        gc.collect()
        torch.cuda.empty_cache()

        logprobs = self.get_logprobs(
            model=model,
            input_ids=query_response,
            attention_mask=attention_mask,
            position_ids=position_ids,
            loss_mask=response_tokens_mask,
        )

        print(f"Logprob from training policy: {logprobs}; Logprob from reference policy: {ref_logprobs}")

        logging.info(f'policy logrobs elapsed time:{time.time() - start}')
        with torch.no_grad():
            kl_term = logprobs.detach() - ref_logprobs
            regularized_rewards = rewards - self.kl_coef * kl_term

            print(f"{regularized_rewards.shape=}", flush=True)
            baselined_reward, baseline_metrics = self.reward_processor.baseline_rewards(rewards=regularized_rewards)

        loss = -baselined_reward * logprobs

        assert len(loss.shape) == 1, loss.shape

        with torch.no_grad():
            metrics = {}
            for tensor, name in zip(
                [
                    rewards,
                    regularized_rewards,
                    baselined_reward,
                    loss.detach(),
                    kl_term.detach(),
                    logprobs.detach(),
                    1 - valid_mask.float(),
                ],
                [
                    'rewards',
                    'regularized_rewards',
                    'baselined_rewards',
                    'loss',
                    'kl_term',
                    'logprobs',
                    'invalid_rewards',
                ],
            ):
                metrics = {
                    **metrics,
                    **get_log_mean_std(tensor.cuda(), name, train_eval),
                }
            metrics[f'{train_eval}/kl_coef'] = self.kl_coef

            for k, v in rewards_metrics.items():
                metrics[f'{train_eval}/{k}'] = v
            for k, v in baseline_metrics.items():
                metrics[f'{train_eval}/{k}'] = v
            
            for k, v in metrics.items():
                if isinstance(v, torch.Tensor):
                    metrics[k] = metrics[k].cpu()
        return loss.mean(), metrics
    
    def print_readable_stats(self):
        if torch.distributed.get_rank() == 0:
            print(f"Allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
            print(f"Reserved: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")

    def compute_loss(self, model, inputs, return_outputs: bool = False, num_items_in_batch=None):
        import logging
        import time
        import gc

        start = time.time()
        
        loss, metrics = self.get_batch_loss_metrics(model, inputs, 'train')

        self.store_metrics(metrics=metrics, train_eval='train')
        logging.info(f'Compute Loss elapsed time:{time.time() - start}')
        gc.collect()
        torch.cuda.empty_cache()

        return (loss, metrics) if return_outputs else loss

    def prediction_step(
        self,
        model: Union[PreTrainedModel, torch.nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        import logging
        logging.info(f'{isinstance(model, DeepSpeedEngine)=}')

        with torch.no_grad():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, 'eval')

        self.store_metrics(metrics, train_eval='eval')

        return loss.detach(), None, None

    @torch.no_grad()
    def reward_stats(
        self, model: torch.nn.Module | PreTrainedModel, dataloader: torch.utils.data.DataLoader
    ) -> tuple[float, float]:
        if self.args.num_samples_for_reward_stats == 0:
            norm_reward_mean = 0
            norm_reward_std = 1
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

            norm_reward_mean = get_global_mean(all_rewards)
            norm_reward_std = get_global_std(all_rewards, mean=norm_reward_mean)

        return norm_reward_mean, norm_reward_std

    def get_logprobs(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        loss_mask: torch.Tensor,
    ):
        from turbo_alignment.trainers.online.ray.rayactor_group import RayGroup
        

        records = {
            'input_ids': input_ids.cuda(),
            'attention_mask': attention_mask.cuda(),
            'position_ids': position_ids.cuda(),
            #'use_cache': False
        }

        lp = None
        if isinstance(model, RayGroup):
            #TODO only one reference model -> maybe delete from group??
            lp = ray.get(model.reference_forward(records, self.args.temperature, loss_mask))
        else:

            hash = 0
            for p in model.parameters():
                hash += p.data.sum().item()
            print("TRAINABLE MODEL HASH: ", hash)

            raw_logits = model(
                input_ids=records['input_ids'],
                attention_mask=records['attention_mask'],
                position_ids=records['position_ids'],
                use_cache=False,
            ).logits[:, :-1] / self.args.temperature

            import logging
            logging.info(f"LOGITS FROM TRAINING POLICY: {raw_logits} ; SUM: {raw_logits.sum()}")

            # Memory efficient - a chain operations
            logits = F.log_softmax(raw_logits, dim=-1)

            # logits /= self.args.temperature
            # all_logprob = F.log_softmax(logits, dim=-1)
            logprob = torch.gather(logits, 2, records['input_ids'][:, 1:].unsqueeze(-1)).squeeze(-1)
            logprob[~loss_mask[:, 1:].to(torch.bool)] = 0

            lp = logprob.sum(-1)

        return lp

    def fill_nonvalid_rewards(self, rewards, query_response) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.args.non_eos_penalty:
            invalid_mask = query_response[:, -1] != self.stop_generation_token_id[0]
            rewards[invalid_mask] = self.args.penalty_reward_value

            return rewards, ~invalid_mask

        return rewards, torch.ones_like(rewards).to(torch.bool)

    def process_rewards(self, rewards, query_response) -> tuple[torch.Tensor, torch.Tensor, Any]:
        """
        Second, this token should be the <eot_id>.
        Otherwise, return a fixed reward of -1.
        """

        rewards, reward_metrics = self.reward_processor.postprocess_rewards(rewards=rewards)
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

    def log(self, logs: Dict[str, float]) -> None:
        main_keys = list(self._stored_metrics.keys())
        for main_key in main_keys:
            for key, metrics in self._stored_metrics[main_key].items():
                logs[key] = torch.tensor(metrics).float().cpu().mean().item()

            if main_key == 'train':
                logs['train/global_step'] = int(self.state.global_step)
            else:
                logs['eval/global_step'] = int(self.state.global_step // self.args.eval_steps)

            del self._stored_metrics[main_key]

        return super().log(logs)  # pylint: disable=no-member
