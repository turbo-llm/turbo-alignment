from typing import Callable

from torch.utils.data import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin

from turbo_alignment.cherry_picks.chat import ChatCherryPickCallback
from turbo_alignment.common.logging import get_project_logger
from turbo_alignment.common.tf.loaders.model.model import load_model
from turbo_alignment.constants import TRAINER_LOGS_FOLDER
from turbo_alignment.dataset.chat.chat import InferenceChatDataset
from turbo_alignment.dataset.kto.collators import KTODataCollator
from turbo_alignment.dataset.loader import DatasetLoader
from turbo_alignment.metrics.metric import Metric
from turbo_alignment.metrics.registry import MetricSettingsRegistry
from turbo_alignment.pipelines.train.base import BaseTrainStrategy
from turbo_alignment.settings.datasets.base import DatasetStrategy
from turbo_alignment.settings.pipelines.train.kto import KTOTrainExperimentSettings
from turbo_alignment.trainers.kto import KTOTrainer, KTOTrainingArguments

logger = get_project_logger()


class TrainKTOStrategy(BaseTrainStrategy[KTOTrainExperimentSettings]):
    @staticmethod
    def _get_data_collator(
        experiment_settings: KTOTrainExperimentSettings,
        tokenizer: PreTrainedTokenizerBase,
        **kwargs,
    ) -> Callable:
        return KTODataCollator(tokenizer=tokenizer, add_labels=True)

    @staticmethod
    def _get_cherry_pick_callback(
        experiment_settings: KTOTrainExperimentSettings,
        tokenizer: PreTrainedTokenizerBase,
        **kwargs,
    ) -> ChatCherryPickCallback:
        cherry_pick_settings = experiment_settings.cherry_pick_settings

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
    def _get_training_args(experiment_settings: KTOTrainExperimentSettings) -> KTOTrainingArguments:
        return KTOTrainingArguments(
            output_dir=str(experiment_settings.log_path / TRAINER_LOGS_FOLDER),
            label_names=[],
            remove_unused_columns=False,
            **experiment_settings.trainer_settings.dict(),
        )

    @staticmethod
    def _get_trainer(
        training_args: KTOTrainingArguments,
        experiment_settings: KTOTrainExperimentSettings,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        train_dataset: Dataset,
        val_dataset: Dataset,
        data_collator: Callable,
    ):
        model.config.use_cache = not experiment_settings.trainer_settings.gradient_checkpointing

        extra_args = {}
        if experiment_settings.trainer_settings.use_ref_model:
            ref_model = load_model(experiment_settings.model_settings, tokenizer)
            for _, param in ref_model.named_parameters():
                param.requires_grad = False

            extra_args = {'ref_model': ref_model}

        return KTOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[],
            data_collator=data_collator,
            tokenizer=tokenizer,
            **extra_args,
        )

    def _dataset_and_collator_sanity_check(self, dataset: Dataset, collator: DataCollatorMixin) -> None:
        logger.info(f'Train sample example:\n{dataset[0]}')
