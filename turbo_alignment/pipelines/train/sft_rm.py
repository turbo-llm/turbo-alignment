from typing import Callable
import torch

import numpy as np
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainingArguments,
    GenerationMixin,
    AutoConfig,
)
from transformers.data.data_collator import DataCollatorMixin

from turbo_alignment.cherry_picks.rm import MultiHeadCherryPickCallback
from turbo_alignment.common.logging import get_project_logger
from turbo_alignment.constants import TRAINER_LOGS_FOLDER
from turbo_alignment.dataset.loader import DatasetLoader
from turbo_alignment.dataset.pair_preferences import (
    PairPreferenceDataCollator,
    PairPreferenceDataset,
)
from turbo_alignment.common.tf.loaders.model.model import load_model
from turbo_alignment.metrics.metric import Metric
from turbo_alignment.metrics.registry import MetricSettingsRegistry
from turbo_alignment.metrics.reward import compute_metrics as compute_rm_metrics
from turbo_alignment.pipelines.train.base import BaseTrainStrategy
from turbo_alignment.settings.datasets import DatasetStrategy
from turbo_alignment.settings.pipelines import RMTrainExperimentSettings
from turbo_alignment.trainers.sft_with_rm import SFTwithRMTrainer

logger = get_project_logger()


class MultiheadModel(PreTrainedModel, GenerationMixin):
    def __init__(self, config, model_settings, tokenizer):
        super().__init__(config)

        self.decoder = load_model(model_settings=model_settings, tokenizer=tokenizer)

        self.lm_head = nn.Linear(self.decoder.norm.weight.shape[0], len(tokenizer), bias=False)
        self.rm_head = nn.Linear(self.decoder.norm.weight.shape[0], 1, bias=False)

        reward_token_ids = tokenizer.encode('<reward>', add_special_tokens=False)
        if len(reward_token_ids) != 1:
            raise ValueError('<reward> token is not found in the tokenizer')

        self.reward_token_ids = reward_token_ids[0]

    def forward(self, batch):
        outputs_w = self.decoder(**batch['inputs_w']).last_hidden_state[0]
        outputs_l = self.decoder(**batch['inputs_l']).last_hidden_state[0]

        reward_token_pos_w = np.where(batch['inputs_w']['input_ids'][0].cpu() == self.reward_token_ids)[0]
        reward_token_pos_l = np.where(batch['inputs_l']['input_ids'][0].cpu() == self.reward_token_ids)[0]

        if len(reward_token_pos_w) != 1 or len(reward_token_pos_l) != 1:
            raise ValueError('More than one <reward> token detected in replica')

        outputs_w_1 = outputs_w[: reward_token_pos_w[0]]
        outputs_w_2 = outputs_w[reward_token_pos_w[0] + 1 :]
        outputs_w_cat = torch.cat((outputs_w_1, outputs_w_2), dim=0)

        lm_logits = self.lm_head(outputs_w_cat)
        rm_logits_w = self.rm_head(outputs_w[reward_token_pos_w[0]])
        rm_logits_l = self.rm_head(outputs_l[reward_token_pos_l[0]])

        return lm_logits, rm_logits_w, rm_logits_l, reward_token_pos_w


class TrainMultiheadStrategy(BaseTrainStrategy[RMTrainExperimentSettings]):
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
    ) -> MultiHeadCherryPickCallback:
        cherry_pick_settings = experiment_settings.cherry_pick_settings

        cherry_pick_datasets = DatasetLoader[PairPreferenceDataset](PairPreferenceDataset).load_datasets(
            cherry_pick_settings.dataset_settings, tokenizer=tokenizer, strategy=DatasetStrategy.TRAIN
        )

        metrics = [
            Metric.by_name(metric.type)(MetricSettingsRegistry.by_name(metric.type)(**metric.parameters))
            for metric in cherry_pick_settings.metric_settings
        ]

        return MultiHeadCherryPickCallback(
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
    def _load_model(
        experiment_settings: RMTrainExperimentSettings,
        tokenizer: PreTrainedTokenizerBase,
    ) -> nn.Module | PreTrainedModel:
        config = AutoConfig.from_pretrained(experiment_settings.model_settings.model_path)
        return MultiheadModel(config, experiment_settings.model_settings, tokenizer)

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
        return SFTwithRMTrainer(
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
