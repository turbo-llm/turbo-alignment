import os

import deepspeed.comm as dist
import pytest
import torch
import torch.distributed
import turbo_alignment.modeling.parallel_states as parallel_states
from transformers import AutoTokenizer, Trainer
from transformers.data.data_collator import default_data_collator

from turbo_alignment.settings.pipelines.train.dpo import DPOTrainExperimentSettings
from turbo_alignment.pipelines.train.dpo import TrainDPOStrategy
from turbo_alignment.common.tf.special_tokens_setter import SpecialTokensSetter
from turbo_alignment.modeling.gemma2.patch import patch_gemma_attn_dict
from turbo_alignment.modeling.mp_pretrained import Gemma2ForCausalLMWithMPU, Gemma2ForCausalLM
from turbo_alignment.modeling.patch_accelerate import patch_acclerator
from turbo_alignment.trainers.base_args import TrainingArgumentsWithSeqP
from turbo_alignment.trainers.dpo import DPOTrainer
from turbo_alignment.dataset.pair_preferences import PairPreferenceDataCollator
from turbo_alignment.modeling.seq_p_collator import pad_for_sequence_parallel, DataCollatorForSequenceParallism

from tests.sequence_parallel.consts import DEEPSPEED_CONFIG
from tests.sequence_parallel.dataset import SimpleDataset
from tests.sequence_parallel.launcher import app
from tests.constants import FIXTURES_PATH
from turbo_alignment.settings.datasets.base import DatasetStrategy

from torch.utils.data import ConcatDataset
from turbo_alignment.dataset.loader import DatasetLoader

SETTINGS_PATH = FIXTURES_PATH / 'configs' / 'train' / 'dpo' / 'dpo_with_seq_p.json'


def run_pipeline(experiment_settings: DPOTrainExperimentSettings):
    with patch_acclerator():
        pipeline = TrainDPOStrategy()
        from turbo_alignment.common import set_random_seed
        set_random_seed(experiment_settings.seed)

        pipeline.tokenizer = pipeline._load_tokenizer(experiment_settings)
        # logger.info('Tokenizer is loaded!')
        pipeline.model = pipeline._load_model(experiment_settings, pipeline.tokenizer)

        training_args = pipeline._get_training_args(experiment_settings)
        print(f'train batch size {training_args.train_batch_size=}')


        # logger.info('Model is loaded!')

        additional_special_tokens = pipeline._get_additional_special_tokens(experiment_settings)
        # logger.info(f'Special tokens: {additional_special_tokens}')
        special_tokens_setter = SpecialTokensSetter(pipeline.tokenizer, experiment_settings.special_tokens_settings)
        special_tokens_setter.set_all()
        special_tokens_setter.set_custom_tokens(additional_special_tokens)

        # logger.info('Special tokens added!')

        # self.model = self._load_model(experiment_settings, self.tokenizer)
        special_tokens_setter.setup_model_config(pipeline.model)

        train_dataset: ConcatDataset = ConcatDataset(
            datasets=DatasetLoader().load_datasets(
                experiment_settings.train_dataset_settings,
                tokenizer=pipeline.tokenizer,
                strategy=DatasetStrategy.TRAIN,
            )
        )

        val_dataset: ConcatDataset = ConcatDataset(
            datasets=DatasetLoader().load_datasets(
                experiment_settings.val_dataset_settings,
                tokenizer=pipeline.tokenizer,
                strategy=DatasetStrategy.TRAIN,
            )
        )

        data_collator = pipeline._get_data_collator(experiment_settings, pipeline.tokenizer)
        if experiment_settings.trainer_settings.sequence_parallel > 1:
            data_collator = DataCollatorForSequenceParallism(
                data_collator,
                seq_p_rank=parallel_states.get_sequence_parallel_rank(),
                seq_p_world_size=parallel_states.get_sequence_parallel_world_size(),
            )

        pipeline.trainer = pipeline._get_trainer(
            training_args,
            experiment_settings,
            pipeline.model,
            pipeline.tokenizer,
            train_dataset,
            val_dataset,
            data_collator,
        )

        set_random_seed(experiment_settings.seed)

        pipeline.trainer.train()


@app.command(name='dpo_model_ulysses')
def dpo_model(
    model_path: str = '/mnt/models/google/gemma-2-2b'
):
    if not os.path.exists(model_path):
        pytest.skip(f'directory {model_path} not found')
        return

    if not os.path.isdir(model_path):
        raise ValueError(f'Model path {model_path} is not a directory')

    patch_gemma_attn_dict()

    experiment_settings = DPOTrainExperimentSettings.parse_file(SETTINGS_PATH)
    run_pipeline(experiment_settings)



@app.command(name='dpo_model_vanilla')
def dpo_model_vanilla(
    model_path: str = '/mnt/models/google/gemma-2-2b'
):
    if not os.path.exists(model_path):
        pytest.skip(f'directory {model_path} not found')
        return

    if not os.path.isdir(model_path):
        raise ValueError(f'Model path {model_path} is not a directory')

    patch_gemma_attn_dict()

    experiment_settings = DPOTrainExperimentSettings.parse_file(SETTINGS_PATH)
    # run_pipeline(experiment_settings)

    vanilla_settings = experiment_settings.copy(deep=True)
    vanilla_settings.model_settings.sequence_parallel_degree = 1
    vanilla_settings.trainer_settings.sequence_parallel = 1
    vanilla_settings.trainer_settings.deepspeed = None

    print(vanilla_settings)

    print('##### RUN VANILLA')
    vanilla_settings.model_settings.model_kwargs["attn_implementation"] = "flash_attention_2"
    vanilla_settings.model_settings.model_type = 'causal'

    run_pipeline(vanilla_settings)


if __name__ == '__main__':
    app()
