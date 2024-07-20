from typing import Callable, cast

from torch.utils.data import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainingArguments
from transformers.data.data_collator import DataCollatorMixin, DataCollatorWithPadding

from turbo_alignment.cherry_picks.classification import ClassificationCherryPickCallback
from turbo_alignment.common.logging import get_project_logger
from turbo_alignment.constants import TRAINER_LOGS_FOLDER
from turbo_alignment.dataset.classification.classification import ClassificationDataset
from turbo_alignment.dataset.loader import DatasetLoader
from turbo_alignment.metrics.metric import Metric
from turbo_alignment.metrics.registry import MetricSettingsRegistry
from turbo_alignment.pipelines.train.base import BaseTrainStrategy
from turbo_alignment.settings.datasets.base import DatasetStrategy
from turbo_alignment.settings.datasets.classification import (
    ClassificationMultiDatasetSettings,
)
from turbo_alignment.settings.pipelines.train.classification import (
    ClassificationTrainExperimentSettings,
)
from turbo_alignment.trainers.classification import (
    ClassificationTrainer,
    auto_class_weights,
)

logger = get_project_logger()


class TrainClassificationStrategy(BaseTrainStrategy[ClassificationTrainExperimentSettings]):
    @staticmethod
    def _get_data_collator(
        experiment_settings: ClassificationTrainExperimentSettings,
        tokenizer: PreTrainedTokenizerBase,
        **_kwargs,
    ) -> Callable:
        dataset_settings: ClassificationMultiDatasetSettings = cast(
            ClassificationMultiDatasetSettings, experiment_settings.train_dataset_settings
        )
        return DataCollatorWithPadding(tokenizer=tokenizer, max_length=dataset_settings.chat_settings.max_tokens_count)

    @staticmethod
    def _get_cherry_pick_callback(
        experiment_settings: ClassificationTrainExperimentSettings,
        tokenizer: PreTrainedTokenizerBase,
        **_kwargs,
    ) -> ClassificationCherryPickCallback:
        cherry_pick_settings = experiment_settings.cherry_pick_settings

        cherry_pick_datasets = DatasetLoader[ClassificationDataset](ClassificationDataset).load_datasets(
            cherry_pick_settings.dataset_settings, tokenizer=tokenizer, strategy=DatasetStrategy.INFERENCE
        )

        metrics = [
            Metric.by_name(metric.type)(MetricSettingsRegistry.by_name(metric.type)(**metric.parameters))
            for metric in cherry_pick_settings.metric_settings
        ]

        return ClassificationCherryPickCallback(
            cherry_pick_settings=cherry_pick_settings,
            datasets=cherry_pick_datasets,
            metrics=metrics,
        )

    @staticmethod
    def _get_training_args(experiment_settings: ClassificationTrainExperimentSettings) -> TrainingArguments:
        return TrainingArguments(
            output_dir=str(experiment_settings.log_path / TRAINER_LOGS_FOLDER),
            label_names=['labels'],
            remove_unused_columns=False,
            **experiment_settings.trainer_settings.dict(exclude={'loss_settings'}),
        )

    @staticmethod
    def _get_trainer(
        training_args: TrainingArguments,
        experiment_settings: ClassificationTrainExperimentSettings,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        train_dataset: Dataset,
        val_dataset: Dataset,
        data_collator: DataCollatorMixin,
    ):
        if experiment_settings.trainer_settings.loss_settings.alpha == 'auto':
            experiment_settings.trainer_settings.loss_settings.alpha = auto_class_weights(train_dataset)
            logger.info(f'Auto computed class weights: {experiment_settings.trainer_settings.loss_settings.alpha}')

        return ClassificationTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            callbacks=[],
            loss_settings=experiment_settings.trainer_settings.loss_settings,
        )

    def _dataset_and_collator_sanity_check(self, dataset: Dataset, collator: DataCollatorMixin) -> None:
        logger.info('Input check: {input_ids}'.format(input_ids=collator([dataset[0], dataset[1]])['input_ids'][0]))

        logger.info('Label check: {labels}'.format(labels=collator([dataset[0], dataset[1]])['labels'][0]))
