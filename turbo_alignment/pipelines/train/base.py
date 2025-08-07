import os
from abc import abstractmethod
import contextlib
from pathlib import Path
from typing import Generic, TypeVar


import torch
from torch.utils.data import ConcatDataset, Dataset
from transformers import (
    PreTrainedTokenizerBase,
    TrainingArguments,
)
from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer import Trainer

from turbo_alignment.cherry_picks.base import CherryPickCallbackBase
from turbo_alignment.common import set_random_seed
from turbo_alignment.common.data.io import write_json
from turbo_alignment.common.logging import get_project_logger
from turbo_alignment.common.tf.loaders.model import load_model
from turbo_alignment.common.tf.loaders.tokenizer import load_tokenizer
from turbo_alignment.common.tf.special_tokens_setter import SpecialTokensSetter
from turbo_alignment.dataset.loader import DatasetLoader
from turbo_alignment.modeling.parallel_states import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
)
from turbo_alignment.pipelines.base import BaseStrategy
from turbo_alignment.pipelines.mixin import S3Mixin
from turbo_alignment.pipelines.mixin.logging import LoggingRegistry
from turbo_alignment.sequence_parallel.collator import DataCollatorForSequenceParallism
from turbo_alignment.sequence_parallel.patch_accelerate import patch_acclerator
from turbo_alignment.settings.datasets.base import DatasetStrategy
from turbo_alignment.settings.pipelines.train.base import BaseTrainExperimentSettings
from turbo_alignment.settings.s3 import ExperimentMetadata, S3HandlerParameters

logger = get_project_logger()


ExperimentSettingsT = TypeVar('ExperimentSettingsT', bound=BaseTrainExperimentSettings)
TrainingArgumentsT = TypeVar('TrainingArgumentsT', bound=TrainingArguments)


class BaseTrainStrategy(S3Mixin, BaseStrategy, Generic[ExperimentSettingsT, TrainingArgumentsT]):
    tokenizer: PreTrainedTokenizerBase
    model: PreTrainedModel
    trainer: Trainer

    @staticmethod
    @abstractmethod
    def _get_cherry_pick_callback(
        experiment_settings: ExperimentSettingsT,
        tokenizer: PreTrainedTokenizerBase,
        **kwargs,
    ) -> CherryPickCallbackBase | None: ...

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
    ) -> DataCollator: ...

    @staticmethod
    @abstractmethod
    def _get_trainer(
        training_args: TrainingArgumentsT,
        experiment_settings: ExperimentSettingsT,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        train_dataset: Dataset,
        val_dataset: Dataset,
        data_collator: DataCollator,
    ) -> Trainer: ...

    @staticmethod
    def _load_model(
        experiment_settings: ExperimentSettingsT, tokenizer: PreTrainedTokenizerBase
    ) -> torch.nn.Module | PreTrainedModel:
        return load_model(experiment_settings.model_settings, tokenizer)

    @staticmethod
    @abstractmethod
    def _get_training_args(experiment_settings: ExperimentSettingsT) -> TrainingArgumentsT: ...

    @staticmethod
    def _load_tokenizer(experiment_settings: ExperimentSettingsT) -> PreTrainedTokenizerBase:
        return load_tokenizer(experiment_settings.tokenizer_settings, experiment_settings.model_settings)

    @abstractmethod
    def _dataset_and_collator_sanity_check(self, dataset: Dataset, collator: DataCollator) -> None: ...

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
        set_random_seed(experiment_settings.seed)

        with patch_acclerator():
            self.tokenizer = self._load_tokenizer(experiment_settings)
            logger.info('Tokenizer is loaded!')
            additional_special_tokens = self._get_additional_special_tokens(experiment_settings)
            logger.info(f'Special tokens: {additional_special_tokens}')
            special_tokens_setter = SpecialTokensSetter(self.tokenizer, experiment_settings.special_tokens_settings)
            special_tokens_setter.set_all()
            special_tokens_setter.set_custom_tokens(additional_special_tokens)

            # In the older version, we loaded the model before args were created,
            # because of bug in embedding resizing with DeepSpeed Zero3
            # Now, we don't observe this bug and this order helps to save RAM

            training_args = self._get_training_args(experiment_settings)

            logger.info('Special tokens added!')
            self.model = self._load_model(experiment_settings, self.tokenizer)  # type: ignore[assignment]
            logger.info('Model is loaded!')

            special_tokens_setter.setup_model_config(self.model)

            train_dataset: ConcatDataset = ConcatDataset(
                datasets=DatasetLoader().load_datasets(
                    experiment_settings.train_dataset_settings,
                    tokenizer=self.tokenizer,
                    strategy=DatasetStrategy.TRAIN,
                    seed=experiment_settings.seed,
                )
            )

            val_dataset: ConcatDataset = ConcatDataset(
                datasets=DatasetLoader().load_datasets(
                    experiment_settings.val_dataset_settings,
                    tokenizer=self.tokenizer,
                    strategy=DatasetStrategy.TRAIN,
                    seed=experiment_settings.seed,
                )
            )

            data_collator = self._get_data_collator(experiment_settings, self.tokenizer)
            if experiment_settings.trainer_settings.sequence_parallel > 1:
                logger.info('Wrap data collator to support sequence parallelism')
                data_collator = DataCollatorForSequenceParallism.create_with_tokenizer(  # type: ignore[assignment]
                    data_collator,
                    seq_p_rank=get_sequence_parallel_rank(),
                    seq_p_world_size=get_sequence_parallel_world_size(),
                    tokenizer=self.tokenizer,
                )

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

            if not self.trainer.args.output_dir:
                raise ValueError('No output dir is set')

            os.makedirs(self.trainer.args.output_dir, exist_ok=True)

            self._save_experiment_config(
                experiment_settings,
                self.trainer.model,  # type: ignore[arg-type]
                Path(self.trainer.args.output_dir) / 'experiment.config',
            )

            experiment_metadata = ExperimentMetadata()
            self._save_experiment_metadata(
                experiment_metadata, Path(self.trainer.args.output_dir) / 'experiment_metadata.json'
            )

            ckpt_dir = '/checkpoints/'
            resume_ckpt = None

            if os.path.isdir(ckpt_dir):
                ckpt_list = [
                    f for f in os.listdir(ckpt_dir) if f.startswith('checkpoint-') and f.split('-')[1].isdigit()
                ]
                if ckpt_list:
                    max_ckpt = max(ckpt_list, key=lambda x: int(x.split('-')[1]))
                    resume_ckpt = os.path.join(ckpt_dir, max_ckpt)
                    print('Возьмем чекпоинт:', resume_ckpt)

            set_random_seed(training_args.seed)

            if resume_ckpt:
                self._patch_trainer_rng_state()
                self.trainer.train(resume_from_checkpoint=resume_ckpt)
            else:
                self.trainer.train()

            self.trainer.save_model()

    @staticmethod
    def _patch_trainer_rng_state() -> None:
        """
        Исправляет Trainer._load_rng_state для PyTorch ≥ 2.6, чтобы
        torch.load(..., weights_only=False).  Накладывается однократно.
        """
        if getattr(Trainer, "_rng_patch_applied", False):
            return

        _orig = Trainer._load_rng_state

        @contextlib.contextmanager
        def _force_weights_only_false():
            real_load = torch.load

            def patched_load(*args, **kwargs):
                kwargs["weights_only"] = False
                return real_load(*args, **kwargs)

            torch.load = patched_load
            try:
                yield
            finally:
                torch.load = real_load

        def _patched(self, checkpoint):
            if checkpoint is None:
                return _orig(self, checkpoint)
            with _force_weights_only_false():
                return _orig(self, checkpoint)

        Trainer._load_rng_state = _patched
        Trainer._rng_patch_applied = True
