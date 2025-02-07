from typing import Callable

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainingArguments
from transformers.data.data_collator import (
    DataCollatorForTokenClassification,
    DataCollatorMixin,
)

from turbo_alignment.cherry_picks.chat import ChatCherryPickCallback
from turbo_alignment.common.logging import get_project_logger
from turbo_alignment.constants import TRAINER_LOGS_FOLDER
from turbo_alignment.dataset.chat import InferenceChatDataset
from turbo_alignment.dataset.loader import DatasetLoader
from turbo_alignment.metrics.metric import Metric
from turbo_alignment.metrics.registry import MetricSettingsRegistry
from turbo_alignment.pipelines.train.base import BaseTrainStrategy
from turbo_alignment.settings.datasets import DatasetStrategy
from turbo_alignment.settings.pipelines.train.sft import SftTrainExperimentSettings
from turbo_alignment.trainers.base_args import TrainingArgumentsWithSeqP
from turbo_alignment.trainers.multigpu import MultiGPUCherryPicksTrainer

logger = get_project_logger()


class DataCollatorForTokenClassificationWithShiftedLabels(DataCollatorForTokenClassification):
    def torch_call(self, features):
        result = super().torch_call(features)
        label_name = 'label' if 'label' in features[0].keys() else 'labels'
        labels = result[label_name]
        last_column = -100 * torch.ones(labels.size(0), 1, dtype=labels.dtype, device=labels.device)
        shifted_labels = torch.cat([labels[:, 1:], last_column], dim=1)
        result[label_name] = shifted_labels
        return result


class TrainSFTStrategy(BaseTrainStrategy[SftTrainExperimentSettings]):
    @staticmethod
    def _get_data_collator(
        experiment_settings: SftTrainExperimentSettings,
        tokenizer: PreTrainedTokenizerBase,
        **kwargs,
    ) -> Callable:
        if experiment_settings.trainer_settings.sequence_parallel == 1:
            return DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)

        return DataCollatorForTokenClassificationWithShiftedLabels(tokenizer, pad_to_multiple_of=8)

    @staticmethod
    def _get_cherry_pick_callback(
        experiment_settings: SftTrainExperimentSettings,
        tokenizer: PreTrainedTokenizerBase,
        **kwargs,
    ) -> ChatCherryPickCallback | None:
        cherry_pick_settings = experiment_settings.cherry_pick_settings
        if cherry_pick_settings is None:
            return None

        cherry_pick_datasets = DatasetLoader[InferenceChatDataset](InferenceChatDataset).load_datasets(
            cherry_pick_settings.dataset_settings, tokenizer=tokenizer, strategy=DatasetStrategy.INFERENCE
        )

        metrics = [
            Metric.by_name(metric.type)(MetricSettingsRegistry.by_name(metric.type)(**metric.parameters))
            for metric in cherry_pick_settings.metric_settings
        ]

        return ChatCherryPickCallback(
            cherry_pick_settings=cherry_pick_settings,
            datasets=cherry_pick_datasets,
            metrics=metrics,
        )

    @staticmethod
    def _get_training_args(experiment_settings: SftTrainExperimentSettings) -> TrainingArguments:
        return TrainingArgumentsWithSeqP(
            output_dir=str(experiment_settings.log_path / TRAINER_LOGS_FOLDER),
            **experiment_settings.trainer_settings.dict(),
        )

    @staticmethod
    def _get_trainer(
        training_args: TrainingArguments,
        experiment_settings: SftTrainExperimentSettings,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        train_dataset: Dataset,
        val_dataset: Dataset,
        data_collator: DataCollatorMixin,
        **_kwargs,
    ) -> MultiGPUCherryPicksTrainer:
        model.config.use_cache = not experiment_settings.trainer_settings.gradient_checkpointing

        return MultiGPUCherryPicksTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[],
            data_collator=data_collator,
            processing_class=tokenizer,
        )

    def _dataset_and_collator_sanity_check(self, dataset: Dataset, collator: DataCollatorMixin) -> None:
        logger.info(f'Train sample example:\n{dataset[0]}')
        logger.info('Example text check: {example}'.format(example=self.tokenizer.decode(dataset[0]['input_ids'])))
        labels_ids = dataset[0]['labels'].clone()
        labels_ids[labels_ids == -100] = 0
        logger.info('Example label text check {example}'.format(example=self.tokenizer.decode(labels_ids)))

        logger.info(
            'Input ids check: {input_ids}'.format(input_ids=collator([dataset[0], dataset[1]])['input_ids'][0])
        )
        logger.info('Mask check: {mask}'.format(mask=collator([dataset[0], dataset[1]])['attention_mask'][0]))
        logger.info('Labels check: {labels}'.format(labels=collator([dataset[0], dataset[1]])['labels'][0]))
