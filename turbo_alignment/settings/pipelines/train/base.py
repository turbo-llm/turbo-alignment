from pathlib import Path
from typing import Any

from pydantic import field_validator
from pydantic_settings import BaseSettings
from transformers import TrainingArguments

from turbo_alignment.common import set_random_seed
from turbo_alignment.constants import TRAINER_LOGS_FOLDER
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


class BaseTrainExperimentSettings(BaseSettings):
    log_path: Path = Path('train_output')
    seed: int = 42

    training_arguments: TrainingArguments

    tokenizer_settings: TokenizerSettings
    special_tokens_settings: SpecialTokensSettings

    model_settings: (ModelForPeftSettings | PreTrainedModelSettings | PreTrainedAdaptersModelSettings)

    train_dataset_settings: MultiDatasetSettings
    val_dataset_settings: MultiDatasetSettings

    logging_settings: (WandbSettings | ClearMLSettings)

    checkpoint_uploader_callback_parameters: CheckpointUploaderCallbackParameters | None = None
    cherry_pick_settings: CherryPickSettings | None = None

    @field_validator('training_arguments', mode='before')
    def create_training_arguments(cls, values: dict[str, Any]) -> TrainingArguments:
        return TrainingArguments(**values, output_dir=TRAINER_LOGS_FOLDER, report_to=[])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.log_path.mkdir(exist_ok=True)
        set_random_seed(self.seed)
        self.training_arguments.output_dir = str(self.log_path / TRAINER_LOGS_FOLDER)
