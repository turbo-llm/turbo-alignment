from typing import Union

from turbo_alignment.settings.cherry_pick import ChatCherryPickSettings
from turbo_alignment.settings.datasets import ChatMultiDatasetSettings
from turbo_alignment.settings.model import (
    PreTrainedAdaptersModelSettings,
    PreTrainedModelSettings,
)
from turbo_alignment.settings.online import (
    ActorType,
    CriticType,
    HFActorSettings,
    RewardProcessorType,
    vLLMActorSettings,
)
from turbo_alignment.settings.pipelines.train.base import BaseTrainExperimentSettings
from turbo_alignment.settings.tf.trainer import TrainerSettings


class REINFORCETrainerSettings(TrainerSettings):
    num_nodes: int = 1
    reward_model_replicas: int = 1
    reference_model_replicas: int = 1
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


class REINFORCETrainExperimentSettings(BaseTrainExperimentSettings):
    train_dataset_settings: ChatMultiDatasetSettings
    val_dataset_settings: ChatMultiDatasetSettings

    reward_model_settings: PreTrainedModelSettings | PreTrainedAdaptersModelSettings

    trainer_settings: REINFORCETrainerSettings
