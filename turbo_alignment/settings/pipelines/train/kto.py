from pydantic import Field

from turbo_alignment.settings.cherry_pick import ChatCherryPickSettings
from turbo_alignment.settings.datasets.kto import KTOMultiDatasetSettings
from turbo_alignment.settings.pipelines.train.base import BaseTrainExperimentSettings
from turbo_alignment.settings.pipelines.train.dpo import SyncRefModelSettings
from turbo_alignment.settings.tf.trainer import TrainerSettings


class KTOTrainerSettings(TrainerSettings):
    undesirable_weight: float = 1.0
    desirable_weight: float = 1.33
    beta: float = 0.1
    use_ref_model: bool = True
    sync_ref_settings: SyncRefModelSettings = SyncRefModelSettings()
    average_log_prob: bool = Field(default=False, description='Normalize log probability by length or not')


class KTOTrainExperimentSettings(BaseTrainExperimentSettings):
    train_dataset_settings: KTOMultiDatasetSettings
    val_dataset_settings: KTOMultiDatasetSettings

    cherry_pick_settings: ChatCherryPickSettings

    trainer_settings: KTOTrainerSettings
