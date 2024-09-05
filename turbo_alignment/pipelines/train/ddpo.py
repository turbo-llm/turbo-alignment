import os
from pathlib import Path
from typing import Callable

from torch.utils.data import ConcatDataset, Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin

from turbo_alignment.cherry_picks.chat import ChatCherryPickCallback
from turbo_alignment.common.logging import get_project_logger
from turbo_alignment.common.tf.callbacks import EvaluateFirstStepCallback
from turbo_alignment.common.tf.loaders.model import load_model
from turbo_alignment.common.tf.loaders.tokenizer import load_tokenizer
from turbo_alignment.common.tf.special_tokens_setter import SpecialTokensSetter
from turbo_alignment.constants import TRAINER_LOGS_FOLDER
from turbo_alignment.dataset.chat.chat import InferenceChatDataset
from turbo_alignment.dataset.ddpo.collators import DDPODataCollator
from turbo_alignment.dataset.ddpo.ddpo import load_ddpo_datasets as load_datasets
from turbo_alignment.dataset.loader import DatasetLoader
from turbo_alignment.metrics.metric import Metric
from turbo_alignment.metrics.registry import MetricSettingsRegistry
from turbo_alignment.pipelines.train.base import BaseTrainStrategy
from turbo_alignment.settings.datasets.base import DatasetStrategy
from turbo_alignment.settings.pipelines.train.ddpo import DDPOTrainExperimentSettings
from turbo_alignment.settings.s3 import ExperimentMetadata
from turbo_alignment.trainers.ddpo import DDPOTrainer, DDPOTrainingArguments

logger = get_project_logger()


class TrainDDPOStrategy(BaseTrainStrategy[DDPOTrainExperimentSettings]):
    rm_tokenizer: PreTrainedTokenizerBase = None
    rm_model: PreTrainedModel = None

    @staticmethod
    def _get_cherry_pick_callback(
        experiment_settings: DDPOTrainExperimentSettings,
        tokenizer: PreTrainedTokenizerBase,
        **kwargs,
    ) -> ChatCherryPickCallback:
        rm_model = kwargs.get('rm_tokenizer', None)
        assert rm_model is not None, 'RM Tokenizer should not be None'

        cherry_pick_settings = experiment_settings.cherry_pick_settings

        cherry_pick_datasets = DatasetLoader[InferenceChatDataset](InferenceChatDataset).load_datasets(
            cherry_pick_settings.dataset_settings, tokenizer=tokenizer, strategy=DatasetStrategy.INFERENCE
        )

        metrics = [
            Metric.by_name(metric.type)(
                MetricSettingsRegistry.by_name(metric.type)(**metric.parameters, rm_model=rm_model)
            )
            for metric in cherry_pick_settings.metric_settings
        ]

        return ChatCherryPickCallback(
            cherry_pick_settings=cherry_pick_settings,
            datasets=cherry_pick_datasets,
            metrics=metrics,
        )

    @staticmethod
    def _get_training_args(experiment_settings: DDPOTrainExperimentSettings) -> DDPOTrainingArguments:
        return DDPOTrainingArguments(
            output_dir=str(experiment_settings.log_path / TRAINER_LOGS_FOLDER),
            label_names=[],
            remove_unused_columns=False,
            beta=experiment_settings.beta,
            use_ref_model=experiment_settings.use_ref_model,
            forward_kl=experiment_settings.forward_kl,
            **experiment_settings.trainer_settings.dict(),
        )

    @staticmethod
    def _get_data_collator(
        experiment_settings: DDPOTrainExperimentSettings, tokenizer: PreTrainedTokenizerBase, **kwargs
    ) -> Callable:
        rm_tokenizer = kwargs.get('rm_tokenizer', None)
        assert rm_tokenizer is not None, 'RM Tokenizer should not be None'
        return DDPODataCollator(
            tokenizer=tokenizer,
            rm_tokenizer=rm_tokenizer,
        )

    @staticmethod
    def _get_trainer(
        training_args: DDPOTrainingArguments,
        experiment_settings: DDPOTrainExperimentSettings,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        train_dataset: Dataset,
        val_dataset: Dataset,
        data_collator: Callable,
        rm_model: PreTrainedModel = None,
    ) -> DDPOTrainer:
        model.config.use_cache = not experiment_settings.trainer_settings.gradient_checkpointing

        extra_args = {'rm': rm_model}

        return DDPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[],
            data_collator=data_collator,
            tokenizer=tokenizer,
            **extra_args,
        )

    def run(self, experiment_settings: DDPOTrainExperimentSettings) -> None:
        training_args = self._get_training_args(experiment_settings)

        self.tokenizer = load_tokenizer(
            experiment_settings.tokenizer_settings,
            experiment_settings.model_settings,
        )

        logger.info('Chat Tokenizer is loaded!')

        self.rm_tokenizer = load_tokenizer(
            experiment_settings.rm_tokenizer_settings,
            experiment_settings.rm_settings,
        )

        logger.info('RM Tokenizer is loaded!')

        additional_special_tokens = self._get_additional_special_tokens(experiment_settings)

        logger.info(f'Special tokens: {additional_special_tokens}')
        special_tokens_setter = SpecialTokensSetter(self.rm_tokenizer, experiment_settings.special_tokens_settings)
        special_tokens_setter.set_all()
        special_tokens_setter.set_custom_tokens(additional_special_tokens)

        logger.info('RM Special tokens added!')

        logger.info(f'Special tokens: {additional_special_tokens}')
        special_tokens_setter = SpecialTokensSetter(self.tokenizer, experiment_settings.special_tokens_settings)
        special_tokens_setter.set_all()
        special_tokens_setter.set_custom_tokens(additional_special_tokens)

        logger.info('Chat Special tokens added!')

        self.model = load_model(experiment_settings.model_settings, self.tokenizer)
        self.rm_model = load_model(experiment_settings.rm_settings, self.rm_tokenizer)

        for _, param in self.rm_model.named_parameters():
            param.requires_grad = False

        logger.info('Model is loaded!')

        train_dataset: ConcatDataset = ConcatDataset(
            datasets=load_datasets(
                experiment_settings.train_dataset_settings,
                rm_tokenizer=self.rm_tokenizer,
                chat_tokenizer=self.tokenizer,
            )
        )
        val_dataset: ConcatDataset = ConcatDataset(
            datasets=load_datasets(
                experiment_settings.val_dataset_settings,
                rm_tokenizer=self.rm_tokenizer,
                chat_tokenizer=self.tokenizer,
            )
        )

        data_collator = self._get_data_collator(
            experiment_settings=experiment_settings, tokenizer=self.tokenizer, rm_tokenizer=self.rm_tokenizer
        )

        self.trainer = self._get_trainer(
            training_args=training_args,
            experiment_settings=experiment_settings,
            model=self.model,
            tokenizer=self.tokenizer,
            rm_model=self.rm_model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            data_collator=data_collator,
        )

        if self.trainer.accelerator.is_main_process:
            self._dataset_and_collator_sanity_check(train_dataset, data_collator)

        self._add_trainer_callbacks(experiment_settings)
        self.trainer.add_callback(EvaluateFirstStepCallback())

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

    def _dataset_and_collator_sanity_check(self, dataset: Dataset, collator: DataCollatorMixin) -> None:
        logger.info(f'Train sample example:\n{dataset[0]}')

        logger.info(
            'SFT Input-w check: {input_ids}'.format(
                input_ids=collator([dataset[0], dataset[1]])['sft_inputs_w']['input_ids'][0]
            )
        )
        logger.info(
            'SFT Mask-w check: {mask}'.format(
                mask=collator([dataset[0], dataset[1]])['sft_inputs_w']['attention_mask'][0]
            )
        )
        logger.info(
            'RM Input-w check: {input_ids}'.format(
                input_ids=collator([dataset[0], dataset[1]])['rm_inputs_w']['input_ids'][0]
            )
        )
        logger.info(
            'RM Mask-w check: {mask}'.format(
                mask=collator([dataset[0], dataset[1]])['rm_inputs_w']['attention_mask'][0]
            )
        )
        logger.info(
            'SFT Input-l check: {input_ids}'.format(
                input_ids=collator([dataset[0], dataset[1]])['sft_inputs_l']['input_ids'][0]
            )
        )
        logger.info(
            'SFT Mask-l check: {mask}'.format(
                mask=collator([dataset[0], dataset[1]])['sft_inputs_l']['attention_mask'][0]
            )
        )

        logger.info(
            'RM Input-l check: {input_ids}'.format(
                input_ids=collator([dataset[0], dataset[1]])['rm_inputs_l']['input_ids'][0]
            )
        )
        logger.info(
            'RM Mask-l check: {mask}'.format(
                mask=collator([dataset[0], dataset[1]])['rm_inputs_l']['attention_mask'][0]
            )
        )
