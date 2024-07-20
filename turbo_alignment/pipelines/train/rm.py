from typing import Callable

from torch.utils.data import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainingArguments
from transformers.data.data_collator import DataCollatorMixin

from turbo_alignment.cherry_picks.rm import RmCherryPickCallback
from turbo_alignment.common.logging import get_project_logger
from turbo_alignment.constants import TRAINER_LOGS_FOLDER
from turbo_alignment.dataset.loader import DatasetLoader
from turbo_alignment.dataset.pair_preferences import (
    PairPreferenceDataCollator,
    PairPreferenceDataset,
)
from turbo_alignment.metrics.metric import Metric
from turbo_alignment.metrics.registry import MetricSettingsRegistry
from turbo_alignment.metrics.reward import compute_metrics as compute_rm_metrics
from turbo_alignment.pipelines.train.base import BaseTrainStrategy
from turbo_alignment.settings.datasets import DatasetStrategy
from turbo_alignment.settings.pipelines import RMTrainExperimentSettings
from turbo_alignment.trainers.rm import RMTrainer

logger = get_project_logger()


class TrainRMStrategy(BaseTrainStrategy[RMTrainExperimentSettings]):
    @staticmethod
    def _get_data_collator(
        experiment_settings: RMTrainExperimentSettings,
        tokenizer: PreTrainedTokenizerBase,
        **_kwargs,
    ) -> Callable:
        return PairPreferenceDataCollator(tokenizer=tokenizer, add_labels=False)

    @staticmethod
    def _get_cherry_pick_callback(
        experiment_settings: RMTrainExperimentSettings,
        tokenizer: PreTrainedTokenizerBase,
        **_kwargs,
    ) -> RmCherryPickCallback:
        cherry_pick_settings = experiment_settings.cherry_pick_settings

        cherry_pick_datasets = DatasetLoader[PairPreferenceDataset](PairPreferenceDataset).load_datasets(
            cherry_pick_settings.dataset_settings, tokenizer=tokenizer, strategy=DatasetStrategy.TRAIN
        )

        metrics = [
            Metric.by_name(metric.type)(MetricSettingsRegistry.by_name(metric.type)(**metric.parameters))
            for metric in cherry_pick_settings.metric_settings
        ]

        return RmCherryPickCallback(
            cherry_pick_settings=cherry_pick_settings,
            datasets=cherry_pick_datasets,
            metrics=metrics,
        )

    @staticmethod
    def _get_training_args(experiment_settings: RMTrainExperimentSettings) -> TrainingArguments:
        return TrainingArguments(
            output_dir=str(experiment_settings.log_path / TRAINER_LOGS_FOLDER),
            label_names=[],
            remove_unused_columns=False,
            **experiment_settings.trainer_settings.dict(),
        )

    @staticmethod
    def _get_trainer(
        training_args: TrainingArguments,
        experiment_settings: RMTrainExperimentSettings,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        train_dataset: Dataset,
        val_dataset: Dataset,
        data_collator: DataCollatorMixin,
        **_kwargs,
    ):
        return RMTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=compute_rm_metrics,
            callbacks=[],
        )

    def _dataset_and_collator_sanity_check(self, dataset: Dataset, collator: DataCollatorMixin) -> None:
        logger.info(f'Train sample input_ids:\n{dataset[0]}')
        logger.info(f'Train sample example:\n{self.tokenizer.decode(dataset[0]["inputs_w"]["input_ids"])}')
        logger.info(f'Train sample example:\n{self.tokenizer.decode(dataset[0]["inputs_l"]["input_ids"])}')
