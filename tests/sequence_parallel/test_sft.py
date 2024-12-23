import enum
import os

from turbo_alignment.modeling.gemma2.patch import patch_gemma_attn_dict
from turbo_alignment.pipelines.train.sft import TrainSFTStrategy
from turbo_alignment.sequence_parallel.patch_accelerate import patch_acclerator
from turbo_alignment.settings.pipelines.train.sft import SftTrainExperimentSettings

from tests.constants import FIXTURES_PATH
from tests.sequence_parallel.launcher import app

SETTINGS_PATH = FIXTURES_PATH / 'configs' / 'train' / 'sft' / 'base_with_seq_p.json'


class AttnType(str, enum.Enum):
    EAGER = 'eager',
    FLASH = 'flash_attention_2'


def run_pipeline(experiment_settings: SftTrainExperimentSettings):
    print(experiment_settings)
    with patch_acclerator():
        pipeline = TrainSFTStrategy()
        pipeline.run(experiment_settings)


@app.command(name='sft_model_ulysses')
def sft_model_ulysses(
    model_path: str = '/mnt/models/google/gemma2-2b',
    attn_type: AttnType = AttnType.EAGER,
):
    if not os.path.exists(model_path):
        return

    if not os.path.isdir(model_path):
        raise ValueError(f'Model path {model_path} is not a directory')

    patch_gemma_attn_dict()

    experiment_settings = SftTrainExperimentSettings.parse_file(SETTINGS_PATH)
    experiment_settings.trainer_settings.load_best_model_at_end = False
    experiment_settings.model_settings.model_kwargs['attn_implementation'] = attn_type.value + '_ulysses'
    run_pipeline(experiment_settings)


@app.command(name='sft_model_vanilla')
def sft_model_vanilla(
    model_path: str = '/mnt/models/google/gemma2-2b',
    attn_type: AttnType = AttnType.EAGER,
):
    if not os.path.exists(model_path):
        return

    if not os.path.isdir(model_path):
        raise ValueError(f'Model path {model_path} is not a directory')

    patch_gemma_attn_dict()

    experiment_settings = SftTrainExperimentSettings.parse_file(SETTINGS_PATH)

    vanilla_settings = experiment_settings.copy(deep=True)
    vanilla_settings.model_settings.sequence_parallel_degree = 1
    vanilla_settings.trainer_settings.sequence_parallel = 1
    vanilla_settings.model_settings.model_kwargs['attn_implementation'] = attn_type.value
    vanilla_settings.model_settings.model_type = 'causal'
    vanilla_settings.trainer_settings.load_best_model_at_end = False

    print('##### RUN VANILLA')
    run_pipeline(vanilla_settings)


def set_prct():
    import prctl
    prctl.set_ptracer(prctl.SET_PTRACER_ANY)


if __name__ == '__main__':
    set_prct()

    import os
    os.register_at_fork(after_in_child=set_prct)
    app()
