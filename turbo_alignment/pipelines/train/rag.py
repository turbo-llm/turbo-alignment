from typing import Callable

import torch
from torch.utils.data import Dataset
from transformers import (
    AdamW,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainingArguments,
)
from transformers.data.data_collator import (
    DataCollatorForTokenClassification,
    DataCollatorMixin,
)

from turbo_alignment.cherry_picks.rag import RagCherryPickCallback
from turbo_alignment.common.logging import get_project_logger
from turbo_alignment.common.tf.loaders.model.model import load_model
from turbo_alignment.constants import TRAINER_LOGS_FOLDER
from turbo_alignment.dataset.chat import InferenceChatDataset
from turbo_alignment.dataset.loader import DatasetLoader
from turbo_alignment.metrics.metric import Metric
from turbo_alignment.metrics.registry import MetricSettingsRegistry
from turbo_alignment.modeling.rag.rag_model import RagSequenceForGeneration
from turbo_alignment.modeling.rag.rag_tokenizer import RagTokenizer
from turbo_alignment.pipelines.train.base import BaseTrainStrategy
from turbo_alignment.settings.datasets import DatasetStrategy
from turbo_alignment.settings.pipelines.train.rag import RAGTrainExperimentSettings
from turbo_alignment.trainers.multigpu import MultiGPUCherryPicksTrainer

logger = get_project_logger()


class TrainRAGStrategy(BaseTrainStrategy[RAGTrainExperimentSettings]):
    @staticmethod
    def _get_data_collator(
        experiment_settings: RAGTrainExperimentSettings,
        tokenizer: PreTrainedTokenizerBase,
        **kwargs,
    ) -> Callable:
        return DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)

    @staticmethod
    def _get_cherry_pick_callback(
        experiment_settings: RAGTrainExperimentSettings,
        tokenizer: PreTrainedTokenizerBase,
        **kwargs,
    ) -> RagCherryPickCallback:
        cherry_pick_settings = experiment_settings.cherry_pick_settings

        cherry_pick_datasets = DatasetLoader[InferenceChatDataset](InferenceChatDataset).load_datasets(
            cherry_pick_settings.dataset_settings, tokenizer=tokenizer, strategy=DatasetStrategy.INFERENCE
        )

        metrics = [
            Metric.by_name(metric.type)(MetricSettingsRegistry.by_name(metric.type)(**metric.parameters))
            for metric in cherry_pick_settings.metric_settings
        ]

        return RagCherryPickCallback(
            cherry_pick_settings=cherry_pick_settings,
            datasets=cherry_pick_datasets,
            metrics=metrics,
        )

    @staticmethod
    def _get_additional_special_tokens(
        experiment_settings: RAGTrainExperimentSettings,  # type: ignore[override]
    ) -> list[str]:
        gen_settings = experiment_settings.model_settings.generator_settings
        embeddings_initialization_strategy = gen_settings.embeddings_initialization_strategy
        return list(embeddings_initialization_strategy.keys()) if embeddings_initialization_strategy else []

    @staticmethod
    def _get_training_args(experiment_settings: RAGTrainExperimentSettings) -> TrainingArguments:
        return TrainingArguments(
            output_dir=str(experiment_settings.log_path / TRAINER_LOGS_FOLDER),
            **experiment_settings.trainer_settings.dict(),
        )

    @staticmethod
    def _get_trainer(
        training_args: TrainingArguments,
        experiment_settings: RAGTrainExperimentSettings,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        train_dataset: Dataset,
        val_dataset: Dataset,
        data_collator: DataCollatorMixin,
        **_kwargs,
    ) -> MultiGPUCherryPicksTrainer:
        trainer = MultiGPUCherryPicksTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[],
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        generator_parameters = model.rag.generator.parameters()
        question_encoder_parameters = model.rag.question_encoder.parameters()
        encoder_lr = experiment_settings.model_settings.retrieval_settings.encoder_learning_rate
        optimizer = AdamW(
            [
                {'params': generator_parameters, 'lr': training_args.learning_rate},
                {
                    'params': question_encoder_parameters,
                    'lr': encoder_lr if encoder_lr is not None else training_args.learning_rate,
                },
            ]
        )
        trainer.optimizer = optimizer
        return trainer

    @staticmethod
    def _load_tokenizer(experiment_settings: RAGTrainExperimentSettings) -> RagTokenizer:
        tokenizer = RagTokenizer(
            model_settings=experiment_settings.model_settings,
            tokenizer_path=experiment_settings.tokenizer_settings.tokenizer_path,
            use_fast=experiment_settings.tokenizer_settings.use_fast,
        )
        return tokenizer

    @staticmethod
    def _load_model(
        experiment_settings: RAGTrainExperimentSettings,
        tokenizer: PreTrainedTokenizerBase,
    ) -> torch.nn.Module | PreTrainedModel:
        model_settings = experiment_settings.model_settings

        generator = load_model(model_settings.generator_settings, tokenizer.generator)
        question_encoder = load_model(model_settings.question_encoder_settings, tokenizer.question_encoder)

        model = RagSequenceForGeneration(model_settings, generator, question_encoder, tokenizer)
        return model

    def _dataset_and_collator_sanity_check(self, dataset: Dataset, collator: DataCollatorMixin) -> None:
        logger.info(f'Train sample example:\n{dataset[0]}')
        logger.info('Example text check: {example}'.format(example=self.tokenizer.decode(dataset[0]['input_ids'])))
        logger.info(
            'Input ids check: {input_ids}'.format(input_ids=collator([dataset[0], dataset[1]])['input_ids'][0])
        )
        logger.info('Mask check: {mask}'.format(mask=collator([dataset[0], dataset[1]])['attention_mask'][0]))
        logger.info('Labels check: {labels}'.format(labels=collator([dataset[0], dataset[1]])['labels'][0]))
