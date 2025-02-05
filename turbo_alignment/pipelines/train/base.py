import os
from abc import abstractmethod
from pathlib import Path
from typing import Callable, Generic, TypeVar

import torch
from torch.utils.data import ConcatDataset, Dataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from turbo_alignment.cherry_picks.base import CherryPickCallbackBase
from turbo_alignment.common.data.io import write_json
from turbo_alignment.common.logging import get_project_logger
from turbo_alignment.common.tf.loaders.model import load_model
from turbo_alignment.common.tf.loaders.tokenizer import load_tokenizer
from turbo_alignment.common.tf.special_tokens_setter import SpecialTokensSetter
from turbo_alignment.dataset.loader import DatasetLoader
from turbo_alignment.pipelines.base import BaseStrategy
from turbo_alignment.pipelines.mixin import S3Mixin
from turbo_alignment.pipelines.mixin.logging import LoggingRegistry
from turbo_alignment.settings.datasets.base import DatasetStrategy
from turbo_alignment.settings.pipelines.train.base import BaseTrainExperimentSettings
from turbo_alignment.settings.s3 import ExperimentMetadata, S3HandlerParameters

logger = get_project_logger()


ExperimentSettingsT = TypeVar('ExperimentSettingsT', bound=BaseTrainExperimentSettings)


class BaseTrainStrategy(S3Mixin, BaseStrategy, Generic[ExperimentSettingsT]):
    tokenizer: PreTrainedTokenizerBase
    model: PreTrainedModel
    trainer: Trainer

    @staticmethod
    @abstractmethod
    def _get_cherry_pick_callback(
        experiment_settings: ExperimentSettingsT,
        tokenizer: PreTrainedTokenizerBase,
        **kwargs,
    ) -> CherryPickCallbackBase | None:
        ...

    @staticmethod
    def _save_experiment_config(
        experiment_settings: ExperimentSettingsT, model: PreTrainedModel, output_path: Path
    ) -> None:
        model_config = model.config.to_dict()
        model_config['experiment_config'] = experiment_settings
        write_json(model_config, output_path)

    @staticmethod
    def _save_experiment_metadata(experiment_metadata: ExperimentMetadata, output_path: Path) -> None:
        write_json(experiment_metadata.dict(), output_path)

    @staticmethod
    @abstractmethod
    def _get_data_collator(
        experiment_settings: ExperimentSettingsT, tokenizer: PreTrainedTokenizerBase, **kwargs
    ) -> Callable:
        ...

    @staticmethod
    @abstractmethod
    def _get_trainer(
        training_args: TrainingArguments,
        experiment_settings: ExperimentSettingsT,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        train_dataset: Dataset,
        val_dataset: Dataset,
        data_collator: Callable,
    ) -> Trainer:
        ...

    @staticmethod
    def _load_model(
        experiment_settings: ExperimentSettingsT, tokenizer: PreTrainedTokenizerBase
    ) -> torch.nn.Module | PreTrainedModel:
        return load_model(experiment_settings.model_settings, tokenizer)

    @staticmethod
    @abstractmethod
    def _get_training_args(experiment_settings: ExperimentSettingsT) -> TrainingArguments:
        ...

    @staticmethod
    def _load_tokenizer(experiment_settings: ExperimentSettingsT) -> PreTrainedTokenizerBase:
        return load_tokenizer(experiment_settings.tokenizer_settings, experiment_settings.model_settings)

    @abstractmethod
    def _dataset_and_collator_sanity_check(self, dataset: Dataset, collator: Callable) -> None:
        ...

    @staticmethod
    def _get_additional_special_tokens(
        experiment_settings: BaseTrainExperimentSettings,
    ) -> list[str]:
        embeddings_initialization_strategy = experiment_settings.model_settings.embeddings_initialization_strategy
        return list(embeddings_initialization_strategy.keys()) if embeddings_initialization_strategy else []

    def _add_trainer_callbacks(self, experiment_settings: ExperimentSettingsT, **kwargs) -> None:
        if self.trainer.accelerator.is_main_process:
            self.trainer.add_callback(
                LoggingRegistry.by_name(experiment_settings.logging_settings.__name__).get_logging_callback(
                    experiment_settings=experiment_settings
                )
            )

        cherry_pick_callback = self._get_cherry_pick_callback(experiment_settings, self.tokenizer, **kwargs)

        if cherry_pick_callback is not None:
            self.trainer.add_callback(cherry_pick_callback)

        if experiment_settings.checkpoint_uploader_callback_parameters is not None:
            if self.trainer.is_deepspeed_enabled and self.trainer.args.deepspeed_plugin.zero_stage == 3:
                raise NotImplementedError(
                    'You should not use checkpoint uploader callback when DeepSpeed ZeRO stage 3 enabled'
                )
            s3_handler = self._get_s3_handler(S3HandlerParameters())

            self.trainer.add_callback(
                self._get_checkpoint_uploader_callback(
                    s3_handler=s3_handler,
                    checkpoint_uploader_callback_parameters=(
                        experiment_settings.checkpoint_uploader_callback_parameters
                    ),
                )
            )

    def run(self, experiment_settings: ExperimentSettingsT) -> None:
        training_args = self._get_training_args(experiment_settings)

        self.tokenizer = self._load_tokenizer(experiment_settings)

        logger.info('Tokenizer is loaded!')

        additional_special_tokens = self._get_additional_special_tokens(experiment_settings)
        logger.info(f'Special tokens: {additional_special_tokens}')
        special_tokens_setter = SpecialTokensSetter(self.tokenizer, experiment_settings.special_tokens_settings)
        special_tokens_setter.set_all()
        special_tokens_setter.set_custom_tokens(additional_special_tokens)

        logger.info('Special tokens added!')

        self.model = self._load_model(experiment_settings, self.tokenizer)

        special_tokens_setter.setup_model_config(self.model)

        logger.info('Model is loaded!')

        train_dataset: ConcatDataset = ConcatDataset(
            datasets=DatasetLoader().load_datasets(
                experiment_settings.train_dataset_settings,
                tokenizer=self.tokenizer,
                strategy=DatasetStrategy.TRAIN,
            )
        )

        val_dataset: ConcatDataset = ConcatDataset(
            datasets=DatasetLoader().load_datasets(
                experiment_settings.val_dataset_settings,
                tokenizer=self.tokenizer,
                strategy=DatasetStrategy.TRAIN,
            )
        )

        data_collator = self._get_data_collator(experiment_settings, self.tokenizer)

        self.trainer = self._get_trainer(
            training_args,
            experiment_settings,
            self.model,
            self.tokenizer,
            train_dataset,
            val_dataset,
            data_collator,
        )

        if self.trainer.accelerator.is_main_process:
            self._dataset_and_collator_sanity_check(train_dataset, data_collator)

        self._add_trainer_callbacks(experiment_settings)

        os.makedirs(self.trainer.args.output_dir, exist_ok=True)
        self._save_experiment_config(
            experiment_settings, self.trainer.model, Path(self.trainer.args.output_dir) / 'experiment.config'
        )

        experiment_metadata = ExperimentMetadata()
        self._save_experiment_metadata(
            experiment_metadata, Path(self.trainer.args.output_dir) / 'experiment_metadata.json'
        )

        self.trainer.train()

        self.trainer.save_model()
