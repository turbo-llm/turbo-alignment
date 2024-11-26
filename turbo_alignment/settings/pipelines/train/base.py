from pathlib import Path

from pydantic_settings import BaseSettings

from turbo_alignment.common import set_random_seed
from turbo_alignment.settings.cherry_pick import CherryPickSettings
from turbo_alignment.settings.datasets.base import MultiDatasetSettings
from turbo_alignment.settings.logging.clearml import ClearMLSettings
from turbo_alignment.settings.logging.weights_and_biases import WandbSettings
from turbo_alignment.settings.model import (
    ModelForPeftSettings,
    PreTrainedAdaptersModelSettings,
    PreTrainedModelSettings,
)
from turbo_alignment.settings.s3 import CheckpointUploaderCallbackParameters
from turbo_alignment.settings.tf.special_tokens_setter import SpecialTokensSettings
from turbo_alignment.settings.tf.tokenizer import TokenizerSettings
from turbo_alignment.settings.tf.trainer import TrainerSettings


class EarlyStoppingSettings(BaseSettings):
    patience: int = 1
    threshold: float | None = 0.0


class BaseTrainExperimentSettings(BaseSettings):
    log_path: Path = Path('train_output')
    seed: int = 42

    trainer_settings: TrainerSettings

    tokenizer_settings: TokenizerSettings
    special_tokens_settings: SpecialTokensSettings

    model_settings: (ModelForPeftSettings | PreTrainedModelSettings | PreTrainedAdaptersModelSettings)

    train_dataset_settings: MultiDatasetSettings
    val_dataset_settings: MultiDatasetSettings

    logging_settings: (WandbSettings | ClearMLSettings)

    checkpoint_uploader_callback_parameters: CheckpointUploaderCallbackParameters | None = None
    cherry_pick_settings: CherryPickSettings | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.log_path.mkdir(exist_ok=True)
        set_random_seed(self.seed)
