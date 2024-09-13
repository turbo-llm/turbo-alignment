from typing import Literal, Optional

from peft import TaskType

from turbo_alignment.settings.base import ExtraFieldsNotAllowedBaseModel
from turbo_alignment.settings.cherry_pick import ChatCherryPickSettings
from turbo_alignment.settings.datasets import ChatMultiDatasetSettings
from turbo_alignment.settings.model import (
    ModelForPeftSettings,
    PreTrainedAdaptersModelSettings,
    PreTrainedModelSettings,
)
from turbo_alignment.settings.pipelines.train.base import BaseTrainExperimentSettings
from turbo_alignment.settings.tf.trainer import TrainerSettings

# @dataclass
# class OnpolicyRuntimeConfig:
#     # various batch sizes
#     world_size: Optional[int] = 2
#     """The number of processes (GPUs) to use"""
#     num_updates: Optional[int] = None
#     """The number of updates to train"""
#     micro_batch_size: Optional[int] = None
#     """The micro batch size across devices (HF's `per_device_train_batch_size` * `world_size`)"""
#     local_batch_size: Optional[int] = None
#     """The batch size per GPU (HF's `per_device_train_batch_size` * `gradient_accumulation_steps`)"""
#     batch_size: Optional[int] = None
#     """The batch size across devices (HF's `per_device_train_batch_size` * `world_size` * `gradient_accumulation_steps`)"""
#     local_mini_batch_size: Optional[int] = None
#     """the mini batch size per GPU"""
#     mini_batch_size: Optional[int] = None
#     """the mini batch size across GPUs"""


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

    num_servers: Optional[int] = None


class PEFTSettings(ExtraFieldsNotAllowedBaseModel):
    inference_mode: bool
    r: int
    lora_alpha: int
    lora_dropout: float
    use_rslora: bool
    init_lora_weights: (
        bool | Literal["gaussian", "pissa", "pissa_niter_[number of iters]", "loftq"]
    )
    task_type: TaskType


class REINFORCETrainExperimentSettings(BaseTrainExperimentSettings):
    train_dataset_settings: ChatMultiDatasetSettings
    val_dataset_settings: ChatMultiDatasetSettings

    cherry_pick_settings: ChatCherryPickSettings

    reward_model_settings: PreTrainedModelSettings | PreTrainedAdaptersModelSettings
    value_model_settings: (
        ModelForPeftSettings | PreTrainedModelSettings | PreTrainedAdaptersModelSettings
    )

    trainer_settings: REINFORCETrainerSettings
    peft_settings: PEFTSettings

    reinforce_class: Literal[
        "REINFORCETrainerVanillaRM",
        "REINFORCETrainerCategoricalPRM",
        "REINFORCETrainerCategoricalPRMVariancePenalty",
        "REINFORCETrainerCategoricalPRMEntropyPenalty",
        "REINFORCETrainerRMRV",
        "REINFORCETrainerRMRVNoEMA",
        "REINFORCETrainerRMRVNoValues",
        "RLOOTrainer",
    ]
