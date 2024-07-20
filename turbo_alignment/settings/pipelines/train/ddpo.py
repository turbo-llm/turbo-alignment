from turbo_alignment.settings.cherry_pick import ChatCherryPickSettings
from turbo_alignment.settings.datasets.ddpo import DDPOMultiDatasetSettings
from turbo_alignment.settings.model import (
    PreTrainedAdaptersModelSettings,
    PreTrainedModelSettings,
)
from turbo_alignment.settings.pipelines.train.base import BaseTrainExperimentSettings
from turbo_alignment.settings.tf.tokenizer import TokenizerSettings


class DDPOTrainExperimentSettings(BaseTrainExperimentSettings):
    beta: float = 0.1
    forward_kl: bool = False

    use_ref_model: bool = True
    rm_settings: PreTrainedModelSettings | PreTrainedAdaptersModelSettings

    train_dataset_settings: DDPOMultiDatasetSettings
    val_dataset_settings: DDPOMultiDatasetSettings

    cherry_pick_settings: ChatCherryPickSettings

    rm_tokenizer_settings: TokenizerSettings
