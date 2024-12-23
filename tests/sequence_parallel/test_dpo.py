import os

from turbo_alignment.modeling.gemma2.patch import patch_gemma_attn_dict
from turbo_alignment.pipelines.train.dpo import TrainDPOStrategy
from turbo_alignment.sequence_parallel.patch_accelerate import patch_acclerator
from turbo_alignment.settings.pipelines.train.dpo import DPOTrainExperimentSettings

from tests.constants import FIXTURES_PATH
from tests.sequence_parallel.launcher import app

SETTINGS_PATH = FIXTURES_PATH / 'configs' / 'train' / 'dpo' / 'dpo_with_seq_p.json'


def run_pipeline(experiment_settings: DPOTrainExperimentSettings):
    print(experiment_settings)
    with patch_acclerator():
        pipeline = TrainDPOStrategy()
        from turbo_alignment.common import set_random_seed
        set_random_seed(experiment_settings.seed)
        pipeline.run(experiment_settings)


@app.command(name='dpo_model_ulysses')
def dpo_model(
    model_path: str = '/mnt/models/google/gemma2-2b'
):
    if not os.path.exists(model_path):
        return

    if not os.path.isdir(model_path):
        raise ValueError(f'Model path {model_path} is not a directory')

    patch_gemma_attn_dict()

    experiment_settings = DPOTrainExperimentSettings.parse_file(SETTINGS_PATH)
    experiment_settings.trainer_settings.load_best_model_at_end = False
    run_pipeline(experiment_settings)


@app.command(name='dpo_model_vanilla')
def dpo_model_vanilla(
    model_path: str = '/mnt/models/google/gemma2-2b'
):
    if not os.path.exists(model_path):
        return

    if not os.path.isdir(model_path):
        raise ValueError(f'Model path {model_path} is not a directory')

    patch_gemma_attn_dict()

    experiment_settings = DPOTrainExperimentSettings.parse_file(SETTINGS_PATH)

    vanilla_settings = experiment_settings.copy(deep=True)
    vanilla_settings.model_settings.sequence_parallel_degree = 1
    vanilla_settings.trainer_settings.sequence_parallel = 1
    vanilla_settings.model_settings.model_kwargs["attn_implementation"] = "eager"
    vanilla_settings.model_settings.model_type = 'causal'
    vanilla_settings.trainer_settings.load_best_model_at_end = False

    print('##### RUN VANILLA')
    run_pipeline(vanilla_settings)


if __name__ == '__main__':
    app()
