from typing import Literal, Optional

from peft import TaskType

from turbo_alignment.settings.cherry_pick import ChatCherryPickSettings
from turbo_alignment.settings.datasets import ChatMultiDatasetSettings
from turbo_alignment.settings.model import (
    PreTrainedAdaptersModelSettings,
    PreTrainedModelSettings,
)
from turbo_alignment.settings.pipelines.train.base import BaseTrainExperimentSettings
from turbo_alignment.settings.tf.trainer import TrainerSettings


class REINFORCETrainerSettings(TrainerSettings):
    max_response_length: Optional[int] = None
    stop_token: Optional[str] = None
    temperature: Optional[float] = None
    penalty_reward_value: Optional[float] = None
    clip_rewards_min: Optional[int] = None
    clip_rewards_max: Optional[int] = None
    """the reward value for responses that do not contain `stop_token_id`"""
    non_eos_penalty: Optional[bool] = None
    """whether to penalize responses that do not contain `stop_token_id`"""

    whiten_rewards: Optional[bool] = None
    """whether to whiten the rewards"""
    kl_coef: Optional[float] = None
    """the KL coefficient"""
    risk_coef: Optional[float] = None
    mean_baseline_coef: Optional[float] = None

    # Adaptive KL rules
    init_kl_coef: Optional[float] = None
    target_kl: Optional[float] = None
    adaptive_kl_k: Optional[float] = None
    adaptive_kl_clip_value: Optional[float] = None
    min_kl_coef: Optional[float] = None
    max_kl_coef: Optional[float] = None

    num_generations: int = 1
    num_samples_for_reward_stats: int = 0


class REINFORCETrainExperimentSettings(BaseTrainExperimentSettings):
    train_dataset_settings: ChatMultiDatasetSettings
    val_dataset_settings: ChatMultiDatasetSettings

    cherry_pick_settings: ChatCherryPickSettings

    reward_model_settings: PreTrainedModelSettings | PreTrainedAdaptersModelSettings

    trainer_settings: REINFORCETrainerSettings
