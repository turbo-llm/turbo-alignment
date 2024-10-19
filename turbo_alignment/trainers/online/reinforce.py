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
)
from turbo_alignment.generators import ChatGenerator, RMSamplingGenerator
from turbo_alignment.generators.base import ChatGeneratorBase
from turbo_alignment.settings.generators.chat import CustomChatGenerationSettings
from turbo_alignment.settings.generators.outputs.chat import ChatInferenceOutput
from turbo_alignment.settings.online import (
    CriticType,
    LLMActorType,
    RewardProcessorType,
)
from turbo_alignment.settings.tf.generation import GeneratorTransformersSettings
from turbo_alignment.trainers.multigpu import MultiGPUCherryPicksTrainer
from turbo_alignment.trainers.online.reward_processor import RewardProcessor
from turbo_alignment.trainers.utils import prepare_model


# FIXME
@dataclass
class REINFORCETrainingArguments(TrainingArguments):
    max_tokens_count: int = 1024
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

    actor_type: LLMActorType = LLMActorType.LOCAL_TRANSFORMERS
    critic_type: CriticType = CriticType.LOCAL_TRANSFORMERS

    reward_processor_type: RewardProcessorType = RewardProcessorType.RLOO


class REINFORCETrainer(MultiGPUCherryPicksTrainer):
    def __init__(
        self,
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
        super().__init__(
            model=policy,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            callbacks=callbacks,
            **kwargs,
        )

        self.ref_model = ref_model
        if self.ref_model is not None:
            self.ref_model = prepare_model(self.ref_model, self.accelerator, self.is_deepspeed_enabled)

        self.reward_model = reward_model

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

        self.generator_transformers_settings = GeneratorTransformersSettings(
            temperature=args.temperature,
            max_length=args.max_tokens_count,
            num_return_sequences=args.num_generations,
            num_beams=args.num_generations,
            do_sample=True,
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

        self.norm_reward_mean, self.norm_reward_std = self.reward_stats(
            model=self.model, dataloader=self.get_train_dataloader()
        )

    # FIXME: some base class instead of RMSamplingGenerator (join with ChatGeneratorBase?)
    def _get_rm_generator(self, reward_model: torch.nn.Module | PreTrainedModel) -> RMSamplingGenerator:
        match self.args.critic_type:
            case CriticType.LOCAL_TRANSFORMERS:
                generator = RMSamplingGenerator(
                    model=reward_model,
                    tokenizer=self.tokenizer,
                    accelerator=self.accelerator,
                )
            case CriticType.DISTRIBUTED_VLLM:
                generator = ...
            case _:
                raise ValueError(f'Critic {self.args.critic_type} is not supported')
        return generator

    def _get_chat_generator(self, model: torch.nn.Module | PreTrainedModel) -> ChatGeneratorBase:
        match self.args.actor_type:
            case LLMActorType.LOCAL_TRANSFORMERS:
                generator = ChatGenerator(
                    model=model,
                    tokenizer=self.tokenizer,
                    transformers_settings=self.generator_transformers_settings,
                    custom_generation_settings=self.generator_custom_settings,
                    accelerator=self.accelerator,
                )
            case LLMActorType.DISTRIBUTED_VLLM:
                generator = ...
            case _:
                raise ValueError(f'LLM Actor {self.args.actor_type} is not supported')
        return generator

    def get_answers_and_rewards(
        self,
        model: torch.nn.Module | PreTrainedModel,
        inputs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        generator = self._get_chat_generator(model)

        generations: list[ChatInferenceOutput] = generator.generate_from_batch_records(
            dataset_name='online', records_batch=inputs
        )

        response_ids = torch.stack([ans.answer_token_ids for g in generations for ans in g.answers])
        response_attention_mask = torch.stack([ans.answer_attention_mask for g in generations for ans in g.answers])

        response_tokens_mask = torch.zeros(response_ids.shape, dtype=torch.float32)
        response_tokens_mask[:, generations[0].input_token_ids.shape[0] :] = 1.0

        position_ids = (response_attention_mask.cumsum(-1) - 1).clamp(min=0)
        position_ids.masked_fill_(response_attention_mask.to(torch.bool) == 0, 0)

        rm_inputs = {
            'input_ids': response_ids,
            'attention_mask': response_attention_mask,
            'position_ids': position_ids,
        }

        critic_generator = self._get_rm_generator(self.reward_model)
        rewards = critic_generator.generate_from_batch_records(rm_inputs)

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
                model=self.accelerator.unwrap_model(model),
                inputs=inputs,
            )

            ref_logprobs = self.get_logprobs(
                model=self.ref_model,
                input_ids=query_response,
                attention_mask=attention_mask,
                position_ids=position_ids,
                loss_mask=response_tokens_mask.to(torch.float32),
            )

            rewards, valid_mask, rewards_metrics = self.process_rewards(rewards=rewards, query_response=query_response)

        logprobs = self.get_logprobs(
            model=model,
            input_ids=query_response,
            attention_mask=attention_mask,
            position_ids=position_ids,
            loss_mask=response_tokens_mask,
        )

        with torch.no_grad():
            kl_term = logprobs.detach() - ref_logprobs
            regularized_rewards = rewards - self.kl_coef * kl_term

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
                    **get_log_mean_std(tensor, name, train_eval),
                }
            metrics[f'{train_eval}/kl_coef'] = self.kl_coef

            for k, v in rewards_metrics.items():
                metrics[f'{train_eval}/{k}'] = v
            for k, v in baseline_metrics.items():
                metrics[f'{train_eval}/{k}'] = v

        return loss.mean(), metrics

    def compute_loss(self, model, inputs, return_outputs: bool = False):
        loss, metrics = self.get_batch_loss_metrics(model, inputs, 'train')

        self.store_metrics(metrics=metrics, train_eval='train')

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
            norm_reward_mean = 0
            norm_reward_std = 1
        else:
            all_rewards = []
            samples_processed = 0
            for batch in dataloader:
                _, _, _, _, rewards = self.get_answers_and_rewards(
                    model=self.accelerator.unwrap_model(model),
                    inputs=batch,
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
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        ).logits[:, :-1]
        logits /= self.args.temperature

        all_logprob = F.log_softmax(logits, dim=-1)

        logprob = torch.gather(all_logprob, 2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        logprob[~loss_mask[:, 1:].to(torch.bool)] = 0

        return logprob.sum(-1)

    def fill_nonvalid_rewards(self, rewards, query_response) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.args.non_eos_penalty:
            assert torch.all(query_response[:, -1] != self.tokenizer.pad_token_id), (
                query_response[:, -1],
                self.tokenizer.pad_token_id,
            )

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
