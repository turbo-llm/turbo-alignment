import logging
import socket
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import ray
import torch
import torch.distributed
import torch.nn.functional as F
import torch.utils
import torch.utils.data
from datasets import Dataset
from deepspeed.runtime.engine import DeepSpeedEngine
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
    TrainingArguments,
)

from turbo_alignment.common.tf.loaders.model.model import disable_dropout_in_model
from turbo_alignment.dist_utils.ray.rayactor_group import RayGroup
from turbo_alignment.dist_utils.ray.vllm_worker_wrap import stateless_init_process_group
from turbo_alignment.dist_utils.utils import get_log_mean_std
from turbo_alignment.generators import ChatGenerator, RMSamplingGenerator
from turbo_alignment.generators.rm import RayRMSamplingGenerator
from turbo_alignment.generators.vllm_chat import RayVLLMChatGenerator
from turbo_alignment.settings.generators.chat import CustomChatGenerationSettings
from turbo_alignment.settings.generators.outputs.chat import ChatInferenceOutput
from turbo_alignment.settings.pipelines.train.online import (
    ActorType,
    CriticType,
    HFActorSettings,
    RewardProcessorType,
    VLLMActorSettings,
)
from turbo_alignment.settings.tf.generation import (
    GeneratorTransformersSettings,
    VLLMGeneratorSettings,
)
from turbo_alignment.trainers.multigpu import MultiGPUCherryPicksTrainer
from turbo_alignment.trainers.reward_processor import RewardProcessor
from turbo_alignment.trainers.utils import prepare_model


# FIXME
@dataclass
class OnlineTrainingArguments(TrainingArguments):
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

    actor_type: ActorType = ActorType.DISTRIBUTED_VLLM
    critic_type: CriticType = CriticType.RAY_TRANSFORMERS
    actor_settings: VLLMActorSettings | HFActorSettings = VLLMActorSettings()

    reward_processor_type: RewardProcessorType = RewardProcessorType.REINFORCE


class OnlineTrainer(MultiGPUCherryPicksTrainer):
    def __init__(
        self,
        vllm_engines: list,
        args: OnlineTrainingArguments,
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

        self.vllm_engines = vllm_engines

        if (
            self.vllm_engines is not None
            and isinstance(args.actor_settings, VLLMActorSettings)
            and torch.distributed.get_rank() == 0
        ):
            master_address = ray._private.services.get_node_ip_address()
            with socket.socket() as sock:
                sock.bind(('', 0))
                master_port = sock.getsockname()[1]

            # TODO_RLOO assert tp_size same for all engines
            # world_size = args.actor_settings['vllm_num_engines']
            #  * args.actor_settings['vllm_tensor_parallel_size'] + 1
            world_size = args.actor_settings.vllm_num_engines * args.actor_settings.vllm_tensor_parallel_size + 1

            refs = [
                engine.init_weight_update_group.remote(
                    master_address,
                    master_port,
                    # i * args.actor_settings['vllm_tensor_parallel_size'] + 1,
                    i * args.actor_settings.vllm_tensor_parallel_size + 1,
                    world_size,
                )
                for i, engine in enumerate(self.vllm_engines)
            ]

            # from turbo_alignment.trainers.online.ray.vllm_worker_wrap import (
            # stateless_init_process_group,
            # )

            # https://github.com/vllm-project/vllm/issues/11399
            # https://github.com/vllm-project/vllm/pull/12084
            # https://github.com/vllm-project/vllm/issues/5723
            self.model_update_group = stateless_init_process_group(
                master_address=master_address,
                master_port=master_port,
                world_size=world_size,
                rank=0,
                device=torch.device('cuda:0'),
            )

            ray.get(refs)

        torch.distributed.barrier()

        self.ref_model = ref_model

        # TODO: TODO_RLOO watch later
        if self.ref_model is not None and not isinstance(self.ref_model, RayGroup):
            self.ref_model = prepare_model(self.ref_model, self.accelerator, self.is_deepspeed_enabled)
            disable_dropout_in_model(self.ref_model)

        elif isinstance(self.ref_model, RayGroup):
            refs = ray.get(self.ref_model.prepare_reference_model(self.accelerator, self.is_deepspeed_enabled))

        disable_dropout_in_model(self.model)

        self.reward_model = reward_model

        self._stored_metrics: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        self.kl_coef = args.kl_coef

        self.gen_id = 1

        # TODO: Take gradient accumulation into account
        # In this way, the effective number of gradient
        # updates taken into account stays the same
        # effective_num_previous_samples = 1 / (1 - self.args.mean_baseline_coef)
        # self.args.mean_baseline_coef =
        # 1 - 1 / (effective_num_previous_samples * self.args.gradient_accumulation_steps)

        self.stop_generation_token_id = processing_class.encode(args.stop_token, add_special_tokens=False)
        assert len(self.stop_generation_token_id) == 1, self.stop_generation_token_id

        # TODO separate stop_strings and eos_token

        self.transformers_settings = GeneratorTransformersSettings(
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            num_return_sequences=args.num_generations,
            num_beams=args.num_generations,
            do_sample=True,
            stop_strings=args.stop_token,
        )

        self.vllm_settings = VLLMGeneratorSettings(
            n=args.num_generations,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            stop_strings=args.stop_token,
        )

        self.generator_custom_settings = CustomChatGenerationSettings(
            batch=self.args.per_device_train_batch_size,
            remove_prompt=False,
        )

        self.reward_processor = RewardProcessor.by_name(args.reward_processor_type)(
            mean_baseline_coef=args.mean_baseline_coef,
            num_generations=args.num_generations,
        )

        self.critic_generator = self._get_rm_generator(self.reward_model)

        # if num_samples_for_reward_stats == 0 then no normalization is done FIXME

        self.norm_reward_mean, self.norm_reward_std = self.reward_stats(
            model=self.model, dataloader=self.get_train_dataloader()
        )

    # pylint: disable=no-member
    # pylint: disable=maybe-no-member
    def _broadcast_to_vllm(self, model: DeepSpeedEngine):
        # avoid OOM
        # torch.cuda.empty_cache()
        model = model.module
        for name, param in model.named_parameters():
            # FIXME i dont know will it work with zero-3
            if torch.distributed.get_rank() == 0:
                refs = [  # pylint: disable=no-member # pylint: disable=maybe-no-member
                    engine.collective_rpc.remote(
                        'update_weight', args=(name, param.dtype, param.shape)
                    )  # pylint: disable=no-member # pylint: disable=maybe-no-member
                    for engine in self.vllm_engines  # pylint: disable=no-member # pylint: disable=maybe-no-member
                ]  # pylint: disable=no-member # pylint: disable=maybe-no-member
                self.model_update_group.broadcast(
                    param, src=0, stream=torch.cuda.current_stream()
                )  # pylint: disable=no-member # pylint: disable=maybe-no-member
                ray.get(refs)

    # FIXME: some base class instead of RMSamplingGenerator (join with ChatGeneratorBase?)
    def _get_rm_generator(
        self, reward_model: torch.nn.Module | PreTrainedModel
    ) -> RMSamplingGenerator | RayRMSamplingGenerator:
        generator: RMSamplingGenerator | RayRMSamplingGenerator

        match self.args.critic_type:
            case CriticType.LOCAL_TRANSFORMERS:
                generator = RMSamplingGenerator(
                    model=reward_model,
                    tokenizer=self.processing_class,
                    micro_batch=1,  # FIXME
                    accelerator=self.accelerator,
                )
            case CriticType.RAY_TRANSFORMERS:
                generator = RayRMSamplingGenerator(
                    model=reward_model,
                    tokenizer=self.processing_class,
                    model_replicas=self.args.reward_model_replicas,
                )  # TODO this type of critic is created for Reward models with CausalLM head and utilize vllm engines
            case CriticType.DISTRIBUTED_VLLM:
                raise ValueError(f'Critic {self.args.critic_type} is not supported yet')
            case _:
                raise ValueError(f'Unknown cricit type {self.args.critic_type}')
        return generator

    # TODO_RLOO why every time generates new object? Since the model weights are changing, recreate new object and
    def _get_chat_generator(
        self, model: torch.nn.Module | PreTrainedModel | None = None
    ) -> ChatGenerator | RayVLLMChatGenerator:
        generator: ChatGenerator | RayVLLMChatGenerator

        match self.args.actor_type:
            case ActorType.LOCAL_TRANSFORMERS:
                generator = ChatGenerator(
                    model=model,
                    tokenizer=self.processing_class,
                    transformers_settings=self.transformers_settings,
                    custom_generation_settings=self.generator_custom_settings,
                    accelerator=self.accelerator,
                )
            case ActorType.DISTRIBUTED_VLLM:
                generator = RayVLLMChatGenerator(
                    vllm_engines=self.vllm_engines,
                    generator_settings=self.vllm_settings,
                    custom_generation_settings=self.generator_custom_settings,
                )
            case _:
                raise ValueError(f'LLM Actor {self.args.actor_type} is not supported')
        return generator

    def get_answers_and_rewards(
        self,
        model: torch.nn.Module | PreTrainedModel | DeepSpeedEngine,
        inputs: dict[str, torch.Tensor],
        do_broadcast=True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if do_broadcast:
            # TODO: move to generator
            torch.distributed.barrier()
            self._broadcast_to_vllm(model)
            torch.distributed.barrier()

        generator = self._get_chat_generator()

        generations: list[ChatInferenceOutput] = generator.generate_from_records(
            records=inputs,
            dataset_name='online',
        )

        response_ids_list: list[torch.Tensor] = [
            torch.cat([ans.input_token_ids, ans.answer_token_ids], dim=1)  # type: ignore[list-item]
            for g in generations
            for ans in g.answers
        ]
        response_attention_mask_list: list[torch.Tensor] = [
            torch.cat([ans.input_attention_mask, ans.answer_attention_mask], dim=1)  # type: ignore[list-item]
            for g in generations
            for ans in g.answers
        ]

        max_length = max(response_id.size(1) for response_id in response_ids_list)

        def pad_sequences(sequences: list[torch.Tensor], max_length: int, pad_value: int = 0) -> torch.Tensor:
            padded_sequences = []
            for seq in sequences:
                padding_needed = max_length - seq.size(1)
                padded_seq = torch.cat(
                    [seq.squeeze(0), torch.full((padding_needed,), pad_value, dtype=seq.dtype, device=seq.device)]
                )
                padded_sequences.append(padded_seq)
            return torch.stack(padded_sequences)

        response_ids: torch.Tensor = pad_sequences(
            response_ids_list, max_length, pad_value=self.tokenizer.pad_token_id
        )
        response_attention_mask: torch.Tensor = pad_sequences(response_attention_mask_list, max_length, pad_value=0)

        del response_ids_list
        del response_attention_mask_list

        response_tokens_mask = torch.zeros(
            response_ids.shape, dtype=torch.bfloat16, device=response_ids.device
        )  # why bf16 FIXME

        input_lengths = [
            ans.input_token_ids.size(1) for g in generations for ans in g.answers  # type: ignore[union-attr]
        ]

        for i, input_len in enumerate(input_lengths):
            response_tokens_mask[i, input_len:] = 1.0  # FIXME

        position_ids = (response_attention_mask.cumsum(-1) - 1).clamp(min=0)
        position_ids.masked_fill_(response_attention_mask.to(torch.bool) == 0, 0).cuda()

        rm_inputs = {
            'input_ids': response_ids,
            'attention_mask': response_attention_mask,
            'position_ids': position_ids,
        }

        rewards = self.critic_generator.generate_from_records(rm_inputs)

        return response_ids, response_attention_mask, response_tokens_mask, position_ids, rewards

    @abstractmethod
    def get_batch_loss_metrics(
        self,
        model: torch.nn.Module | PreTrainedModel,
        inputs: dict[str, torch.Tensor],
        train_eval: Literal['train', 'eval'] = 'train',
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        ...

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,  # pylint: disable=unused-argument
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        loss, metrics = self.get_batch_loss_metrics(model, inputs, 'train')

        self.store_metrics(metrics=metrics, train_eval='train')

        # gc.collect()
        # torch.cuda.empty_cache() #NEED?

        if return_outputs:
            return loss, metrics

        return loss

    def prediction_step(
        self,
        model: Union[PreTrainedModel, torch.nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if ignore_keys is None:
            if hasattr(model, 'config'):
                ignore_keys = getattr(model.config, 'keys_to_ignore_at_inference', [])
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, 'eval')

        self.store_metrics(metrics, train_eval='eval')

        if prediction_loss_only:
            return loss.detach(), None, None

        logits = torch.zeros(loss.shape[0])  # FIXME
        labels = torch.zeros(loss.shape[0])

        return loss.detach(), logits, labels

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
                _, _, _, _, rewards = self.get_answers_and_rewards(model=model, inputs=batch, do_broadcast=False)
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
        if self.args.non_eos_penalty:
            invalid_mask = query_response[:, -1] != self.stop_generation_token_id[0]
            rewards[invalid_mask] = self.args.penalty_reward_value

            return rewards, ~invalid_mask

        return rewards, torch.ones_like(rewards).to(torch.bool)

    def process_rewards(self, rewards, query_response) -> tuple[torch.Tensor, torch.Tensor, Any]:  # bottleneck
        """
        Second, this token should be the <eot_id>.
        Otherwise, return a fixed reward of -1.
        """

        rewards, reward_metrics = self.reward_processor.postprocess_rewards(rewards=rewards)  # FIXME 4 secods?
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

    def log(self, logs: Dict[str, float], _start_time: float | None = None) -> None:
        main_keys = list(self._stored_metrics.keys())
        for main_key in main_keys:
            for key, metrics in self._stored_metrics[main_key].items():
                if not isinstance(metrics, (list, tuple)):
                    metrics = [metrics]

                float_values = []
                tensor_values: list[torch.Tensor] = []
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
