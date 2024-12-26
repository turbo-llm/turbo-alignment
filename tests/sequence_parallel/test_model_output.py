import os
import pathlib
import subprocess

import pytest

from tests.constants import FIXTURES_PATH
from tests.sequence_parallel.compare_gradients import compare as compare_gradients
from tests.sequence_parallel.compare_values import compare as compare_values

DPO_SETTINGS_PATH = FIXTURES_PATH / 'configs' / 'train' / 'dpo' / 'dpo_with_seq_p.json'
SFT_SETTINGS_PATH = FIXTURES_PATH / 'configs' / 'train' / 'sft' / 'base_with_seq_p.json'
SFT_PEFT_SETTINGS_PATH = FIXTURES_PATH / 'configs' / 'train' / 'sft' / 'peft_with_seq_p.json'

DIR = pathlib.Path(__file__).parent


@pytest.mark.parametrize(
    'task_type,settings_path',
    [
        pytest.param('dpo', DPO_SETTINGS_PATH, id='dpo'),
        pytest.param('sft', SFT_SETTINGS_PATH, id='sft'),
        pytest.param('sft', SFT_PEFT_SETTINGS_PATH, id='sft_lora', marks=pytest.mark.xfail),
    ],
)
def test_model_output(task_type: str, settings_path: pathlib.Path, tmp_path_factory: pytest.TempPathFactory):
    env = os.environ

    script_path = str((DIR / 'test_dpo.py').absolute())

    gradient_dir = tmp_path_factory.mktemp('gradient').absolute()
    forward_dir = tmp_path_factory.mktemp('forward').absolute()

    env['GRADIENT_HOOK_OUTPUT_DIR'] = str(gradient_dir)
    env['FORWARD_HOOK_OUTPUT_DIR'] = str(forward_dir)
    env['CREATE_GRADIENT_HOOK'] = '1'

    common_cmd_line = [
        'deepspeed',
        '--no_local_rank',
    ]

    script_args = [
        script_path,
        '--task-type',
        task_type,
        '--settings-path',
        str(settings_path.absolute()),
    ]

    print('Run with ulessys')
    subprocess.check_call(common_cmd_line + ['--num_gpus','2'] + script_args)

    print('Run vanilla')
    subprocess.check_call(common_cmd_line + ['--num_gpus','1'] + script_args + ['--make-model-vanilla'])

    compare_values(forward_dir)
    compare_gradients(gradient_dir)
