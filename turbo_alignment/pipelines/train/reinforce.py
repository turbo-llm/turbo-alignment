from typing import Callable

from torch.utils.data import ConcatDataset, Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin

import turbo_alignment.trainers.reinforce as reinforce_trainers
from turbo_alignment.cherry_picks.chat import ChatCherryPickCallback
from turbo_alignment.common.logging import get_project_logger
from turbo_alignment.constants import TRAINER_LOGS_FOLDER
from turbo_alignment.dataset.chat.chat import InferenceChatDataset
from turbo_alignment.dataset.chat.collators import REINFORCEDataCollator
from turbo_alignment.dataset.loader import DatasetLoader
from turbo_alignment.metrics.metric import Metric
from turbo_alignment.metrics.registry import MetricSettingsRegistry
from turbo_alignment.pipelines.train.base import BaseTrainStrategy
from turbo_alignment.settings.datasets.base import DatasetStrategy
from turbo_alignment.settings.pipelines.train.reinforce import (
    REINFORCETrainerSettings,
    REINFORCETrainExperimentSettings,
)
from turbo_alignment.trainers.reinforce import REINFORCETrainingArguments

logger = get_project_logger()


class TrainREINFORCEStrategy(BaseTrainStrategy[REINFORCETrainExperimentSettings]):
    @staticmethod
    def _get_data_collator(
        experiment_settings: REINFORCETrainExperimentSettings,
        tokenizer: PreTrainedTokenizerBase,
        **kwargs,
    ) -> Callable:
        return REINFORCEDataCollator(tokenizer)

    @staticmethod
    def _get_cherry_pick_callback(
        experiment_settings: REINFORCETrainExperimentSettings,
        tokenizer: PreTrainedTokenizerBase,
        **kwargs,
    ) -> ChatCherryPickCallback:
        cherry_pick_settings = experiment_settings.cherry_pick_settings

        cherry_pick_datasets = DatasetLoader[InferenceChatDataset](
            InferenceChatDataset
        ).load_datasets(
            cherry_pick_settings.dataset_settings,
            tokenizer=tokenizer,
            strategy=DatasetStrategy.INFERENCE,
        )

        metrics = [
            Metric.by_name(metric.type)(
                MetricSettingsRegistry.by_name(metric.type)(**metric.parameters)
            )
            for metric in cherry_pick_settings.metric_settings
        ]

        return ChatCherryPickCallback(
            cherry_pick_settings=cherry_pick_settings,
            datasets=cherry_pick_datasets,
            metrics=metrics,
        )

    @staticmethod
    def _get_training_args(
        experiment_settings: REINFORCETrainExperimentSettings,
    ) -> REINFORCETrainerSettings:
        return REINFORCETrainingArguments(
            output_dir=str(experiment_settings.log_path / TRAINER_LOGS_FOLDER),
            label_names=[],
            remove_unused_columns=False,
            report_to=[],
            **experiment_settings.trainer_settings.dict(),
        )

    @staticmethod
    def _get_trainer(
        training_args: REINFORCETrainerSettings,
        experiment_settings: REINFORCETrainExperimentSettings,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        train_dataset: Dataset,
        val_dataset: Dataset,
        data_collator: Callable,
    ):
        cls = getattr(reinforce_trainers, experiment_settings.reinforce_class)

        return cls(
            args=training_args,
            tokenizer=tokenizer,
            policy=model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            callbacks=[],
            peft_config=experiment_settings.peft_settings.dict(),
            policy_model_dir=experiment_settings.model_settings.model_path,
            max_tokens_count=experiment_settings.train_dataset_settings.max_tokens_count,
        )

    def _dataset_and_collator_sanity_check(
        self, dataset: Dataset, collator: DataCollatorMixin
    ) -> None:
        logger.info(f"Train sample input_ids:\n{dataset[0]}")
        logger.info(
            f'Train sample example:\n{self.tokenizer.decode(dataset[0]["input_ids"])}'
        )

    def _get_datasets(
        self, experiment_settings: REINFORCETrainExperimentSettings
    ) -> tuple[Dataset, Dataset]:
        train_dataset: ConcatDataset = ConcatDataset(
            datasets=DatasetLoader[InferenceChatDataset](
                InferenceChatDataset
            ).load_datasets(
                experiment_settings.train_dataset_settings,
                tokenizer=self.tokenizer,
                strategy=DatasetStrategy.INFERENCE,
            )
        )

        val_dataset: ConcatDataset = ConcatDataset(
            datasets=DatasetLoader[InferenceChatDataset](
                InferenceChatDataset
            ).load_datasets(
                experiment_settings.val_dataset_settings,
                tokenizer=self.tokenizer,
                strategy=DatasetStrategy.INFERENCE,
            )
        )
        return train_dataset, val_dataset
