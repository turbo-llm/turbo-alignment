from typing import Callable

import torch.nn
from torch.utils.data import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainingArguments
from transformers.data.data_collator import DataCollatorMixin

from turbo_alignment.cherry_picks.multimodal import MultimodalCherryPickCallback
from turbo_alignment.common.logging import get_project_logger
from turbo_alignment.common.tf.loaders import load_model
from turbo_alignment.constants import TRAINER_LOGS_FOLDER
from turbo_alignment.dataset.loader import DatasetLoader
from turbo_alignment.dataset.multimodal import InferenceMultimodalDataset
from turbo_alignment.dataset.multimodal.collators import DataCollatorWithModalityInputs
from turbo_alignment.metrics.metric import Metric
from turbo_alignment.metrics.registry import MetricSettingsRegistry
from turbo_alignment.modeling.multimodal.lm.projection import (
    ProjectionMultiModalModeling,
)
from turbo_alignment.pipelines.mixin.multimodal import MultimodalMixin
from turbo_alignment.pipelines.train.base import BaseTrainStrategy
from turbo_alignment.settings.datasets import DatasetStrategy
from turbo_alignment.settings.pipelines.train.multimodal import (
    MultimodalTrainExperimentSettings,
)
from turbo_alignment.trainers.multimodal import TrainerCustomSave

logger = get_project_logger()


class TrainMultimodalStrategy(MultimodalMixin, BaseTrainStrategy[MultimodalTrainExperimentSettings]):
    @staticmethod
    def _get_data_collator(
        experiment_settings: MultimodalTrainExperimentSettings,
        tokenizer: PreTrainedTokenizerBase,
        **kwargs,
    ) -> Callable:
        return DataCollatorWithModalityInputs(tokenizer, pad_to_multiple_of=8)

    @staticmethod
    def _get_cherry_pick_callback(
        experiment_settings: MultimodalTrainExperimentSettings,
        tokenizer: PreTrainedTokenizerBase,
        **kwargs,
    ) -> MultimodalCherryPickCallback:
        cherry_pick_settings = experiment_settings.cherry_pick_settings

        cherry_pick_datasets = DatasetLoader[InferenceMultimodalDataset](InferenceMultimodalDataset).load_datasets(
            cherry_pick_settings.dataset_settings, tokenizer=tokenizer, strategy=DatasetStrategy.INFERENCE
        )

        metrics = [
            Metric.by_name(metric.type)(MetricSettingsRegistry.by_name(metric.type)(**metric.parameters))
            for metric in cherry_pick_settings.metric_settings
        ]

        return MultimodalCherryPickCallback(
            cherry_pick_settings=cherry_pick_settings,
            datasets=cherry_pick_datasets,
            metrics=metrics,
        )

    @staticmethod
    def _get_training_args(experiment_settings: MultimodalTrainExperimentSettings) -> TrainingArguments:
        return TrainingArguments(
            output_dir=str(experiment_settings.log_path / TRAINER_LOGS_FOLDER),
            **experiment_settings.trainer_settings.dict(),
        )

    @staticmethod
    def _get_trainer(
        training_args: TrainingArguments,
        experiment_settings: MultimodalTrainExperimentSettings,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        train_dataset: Dataset,
        val_dataset: Dataset,
        data_collator: DataCollatorMixin,
        **_kwargs,
    ):
        return TrainerCustomSave(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[],
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

    @staticmethod
    def _load_model(
        experiment_settings: MultimodalTrainExperimentSettings,
        tokenizer: PreTrainedTokenizerBase,
    ) -> torch.nn.Module | PreTrainedModel:
        language_model = load_model(experiment_settings.model_settings, tokenizer)

        modality_encoders = TrainMultimodalStrategy._load_modality_encoders(
            experiment_settings.modality_encoder_settings_mapping,
            device=language_model.device,
            dtype=language_model.dtype,
        )

        return ProjectionMultiModalModeling(
            language_model=language_model,
            encoders=modality_encoders,
            n_modality_embs=experiment_settings.train_dataset_settings.n_modality_embeddings,
            modality_projector_mapping=experiment_settings.modality_projector_mapping,
            modality_projector_initialization_mapping=experiment_settings.modality_projector_initialization_mapping,
            peft=True,
        )

    def _dataset_and_collator_sanity_check(self, dataset: Dataset, collator: DataCollatorMixin) -> None:
        logger.info(f'Train sample example:\n{dataset[0]}')
        logger.info(
            'Input ids check: {input_ids}'.format(input_ids=collator([dataset[0], dataset[1]])['input_ids'][0])
        )
        logger.info('Mask check: {mask}'.format(mask=collator([dataset[0], dataset[1]])['attention_mask'][0]))
        logger.info('Labels check: {labels}'.format(labels=collator([dataset[0], dataset[1]])['labels'][0]))
