import os
from pathlib import Path
from typing import Callable, TypeVar

import ray
from torch.utils.data import ConcatDataset, Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin

from turbo_alignment.common.logging import get_project_logger
from turbo_alignment.common.tf.loaders import load_model
from turbo_alignment.common.tf.special_tokens_setter import SpecialTokensSetter
from turbo_alignment.constants import TRAINER_LOGS_FOLDER
from turbo_alignment.dataset.chat.chat import InferenceChatDataset
from turbo_alignment.dataset.loader import DatasetLoader
from turbo_alignment.pipelines.train.base import BaseTrainStrategy
from turbo_alignment.pipelines.train.reinforce import ReinforceDataCollator
from turbo_alignment.settings.datasets.base import DatasetStrategy
from turbo_alignment.settings.pipelines.train.base import BaseTrainExperimentSettings
from turbo_alignment.settings.pipelines.train.reinforce import (
    REINFORCETrainExperimentSettings,
)
from turbo_alignment.settings.s3 import ExperimentMetadata
from turbo_alignment.trainers.online.grpo import GRPOTrainer
from turbo_alignment.trainers.online.ray.distributed_torch_ray_actor import (
    DistributedTorchRayActor,
)
from turbo_alignment.trainers.online.reinforce import REINFORCETrainingArguments

ExperimentSettingsT = TypeVar('ExperimentSettingsT', bound=BaseTrainExperimentSettings)

logger = get_project_logger()


@ray.remote(num_gpus=1)
class TrainGRPOStrategy(BaseTrainStrategy[REINFORCETrainExperimentSettings], DistributedTorchRayActor):
    def __init__(self, world_size, rank, local_rank, master_addr, master_port):
        super().__init__(
            world_size=world_size, rank=rank, local_rank=local_rank, master_addr=master_addr, master_port=master_port
        )
        self.node_id = ray.get_runtime_context().get_node_id()
        self.local_rank = ray.get_gpu_ids()

    def init_model_from_pretrained(self):
        self._setup_distributed()

    @staticmethod
    def _get_data_collator(
        experiment_settings: REINFORCETrainExperimentSettings,
        tokenizer: PreTrainedTokenizerBase,
        **kwargs,
    ) -> Callable:
        return ReinforceDataCollator(tokenizer, pad_to_multiple_of=8)

    @staticmethod
    def _get_cherry_pick_callback(
        experiment_settings: REINFORCETrainExperimentSettings,
        tokenizer: PreTrainedTokenizerBase,
        **kwargs,
    ) -> None:
        return None

    @staticmethod
    def _get_training_args(
        experiment_settings: REINFORCETrainExperimentSettings,
    ) -> REINFORCETrainingArguments:
        return REINFORCETrainingArguments(
            output_dir=str(experiment_settings.log_path / TRAINER_LOGS_FOLDER),
            label_names=[],
            remove_unused_columns=False,
            **experiment_settings.trainer_settings.dict(),
        )

    # TODO: TODO_RLOO delete reference and reward model
    @staticmethod
    def _get_trainer(
        vllm_engines,
        training_args: REINFORCETrainingArguments,
        experiment_settings: REINFORCETrainExperimentSettings,
        reward_model,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        train_dataset: Dataset,
        val_dataset: Dataset,
        data_collator: Callable,
        ref_model=None,
    ):
        # TODO: different tokenizer for reward model

        # TODO: TODO_RLOO load reference and reward model here

        if ref_model is None:
            ref_model = load_model(experiment_settings.model_settings, tokenizer)
            for _, param in ref_model.named_parameters():
                param.requires_grad = False

            ref_model.eval()

        # reward_model = load_model(experiment_settings.reward_model_settings, tokenizer)
        # for _, param in reward_model.named_parameters():
        #     param.requires_grad = False

        # reward_model.eval()

        return GRPOTrainer(
            vllm_engines=vllm_engines,
            args=training_args,
            processing_class=tokenizer,
            policy=model,
            ref_model=ref_model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            reward_model=reward_model,
            callbacks=[],
        )

    def _dataset_and_collator_sanity_check(self, dataset: Dataset, collator: DataCollatorMixin) -> None:
        logger.info(f'Train sample input_ids:\n{dataset[0]}')
        logger.info(f'Train sample example:\n{self.tokenizer.decode(dataset[0]["input_ids"])}')

    def _get_datasets(self, experiment_settings: REINFORCETrainExperimentSettings) -> tuple[Dataset, Dataset]:
        train_dataset: ConcatDataset = ConcatDataset(
            datasets=DatasetLoader[InferenceChatDataset](InferenceChatDataset).load_datasets(
                experiment_settings.train_dataset_settings,
                tokenizer=self.tokenizer,
                strategy=DatasetStrategy.INFERENCE,
            )
        )

        val_dataset: ConcatDataset = ConcatDataset(
            datasets=DatasetLoader[InferenceChatDataset](InferenceChatDataset).load_datasets(
                experiment_settings.val_dataset_settings,
                tokenizer=self.tokenizer,
                strategy=DatasetStrategy.INFERENCE,
            )
        )
        return train_dataset, val_dataset

    # TODO
    # get rid off vllm_engines, reference_model, reward_model if possible
    # only get_trainer affected

    def run(self, experiment_settings: ExperimentSettingsT, vllm_engines, reward_model) -> None:  # reference_model
        training_args = self._get_training_args(experiment_settings)

        # import torch
        # torch.cuda.memory._record_memory_history()
        self.tokenizer = self._load_tokenizer(experiment_settings)

        logger.info('Tokenizer is loaded!')

        additional_special_tokens = self._get_additional_special_tokens(experiment_settings)
        logger.info(f'Special tokens: {additional_special_tokens}')
        special_tokens_setter = SpecialTokensSetter(self.tokenizer, experiment_settings.special_tokens_settings)
        special_tokens_setter.set_all()
        special_tokens_setter.set_custom_tokens(additional_special_tokens)

        logger.info('Special tokens added!')

        import time

        start = time.time()

        self.model = self._load_model(experiment_settings, self.tokenizer)

        print(f'Elapsed model load time: {time.time() - start} seconds')

        special_tokens_setter.setup_model_config(self.model)

        logger.info('Model is loaded!')

        train_dataset: ConcatDataset = ConcatDataset(
            datasets=DatasetLoader().load_datasets(
                experiment_settings.train_dataset_settings,
                tokenizer=self.tokenizer,
                strategy=DatasetStrategy.INFERENCE,
            )
        )

        val_dataset: ConcatDataset = ConcatDataset(
            datasets=DatasetLoader().load_datasets(
                experiment_settings.val_dataset_settings,
                tokenizer=self.tokenizer,
                strategy=DatasetStrategy.INFERENCE,
            )
        )

        data_collator = self._get_data_collator(experiment_settings, self.tokenizer)

        start = time.time()

        self.trainer = self._get_trainer(
            vllm_engines,
            training_args,
            experiment_settings,
            # reference_model,
            reward_model,
            self.model,
            self.tokenizer,
            train_dataset,
            val_dataset,
            data_collator,
        )
        print(f'Elapsed get_trainer time: {time.time() - start} seconds')

        if self.trainer.accelerator.is_main_process:
            self._dataset_and_collator_sanity_check(train_dataset, data_collator)

        self._add_trainer_callbacks(experiment_settings)

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
