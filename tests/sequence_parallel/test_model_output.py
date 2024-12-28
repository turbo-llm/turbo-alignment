import json
import os
import pathlib
import subprocess
from typing import Any

import numpy as np
import pytest

from tests.constants import FIXTURES_PATH
from tests.sequence_parallel.consts import MODEL_PATH
from tests.sequence_parallel.utils import read_first_line
from tests.sequence_parallel.compare_gradients import compare as compare_gradients
from tests.sequence_parallel.compare_values import compare as compare_values
from tests.sequence_parallel.marks import has_gemma_model, has_two_gpus

DPO_SETTINGS_PATH = FIXTURES_PATH / 'configs' / 'train' / 'dpo' / 'dpo_with_seq_p.json'
SFT_SETTINGS_PATH = FIXTURES_PATH / 'configs' / 'train' / 'sft' / 'base_with_seq_p.json'
SFT_PEFT_SETTINGS_PATH = FIXTURES_PATH / 'configs' / 'train' / 'sft' / 'peft_with_seq_p.json'

DIR = pathlib.Path(__file__).parent


def patch_settings(settings: dict[str, Any], checkpoint_dir: pathlib.Path, model_path: str) -> dict[str, Any]:
    settings = settings.copy()
    settings['model_settings']['model_path'] = model_path
    settings['log_path'] = str(checkpoint_dir.absolute())
    return settings


@pytest.mark.parametrize(
    'task_type,settings_path',
    [
        pytest.param('dpo', DPO_SETTINGS_PATH, id='dpo'),
        pytest.param('sft', SFT_SETTINGS_PATH, id='sft'),
        pytest.param('sft', SFT_PEFT_SETTINGS_PATH, id='sft_lora', marks=pytest.mark.xfail),
    ],
)
@pytest.mark.skipif(not has_two_gpus(), reason='At least two gpu are required')
@pytest.mark.skipif(not has_gemma_model(), reason='Gemma model not found')
def test_model_output(task_type: str, settings_path: pathlib.Path, tmp_path_factory: pytest.TempPathFactory):
    env = os.environ

    script_path = str((DIR / 'test_dpo.py').absolute())

    gradient_dir = tmp_path_factory.mktemp('gradient').absolute()
    forward_dir = tmp_path_factory.mktemp('forward').absolute()
    checkpoint_dir = tmp_path_factory.mktemp('checkpoint').absolute()
    settings_dir = tmp_path_factory.mktemp('settings').absolute()
    new_settings_path = settings_dir / 'settings.json'

    with settings_path.open('r', encoding='utf-8') as input_:
        settings = json.load(input_)

    new_settings = patch_settings(settings, checkpoint_dir, MODEL_PATH)

    with new_settings_path.open('w', encoding='utf-8') as output:
        json.dump(new_settings, output, indent=4, ensure_ascii=False)

    print(new_settings)

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
        str(new_settings_path.absolute()),
    ]

    print('Run with ulessys')
    subprocess.check_call(common_cmd_line + ['--num_gpus','2'] + script_args)
    print('Run vanilla')
    subprocess.check_call(common_cmd_line + ['--num_gpus','1'] + script_args + ['--make-model-vanilla'])

    attention_mask_shape_file = forward_dir / 'attention_mask.shape'
    attention_mask = None
    if attention_mask_shape_file.exists():
        shape = tuple(map(int, read_first_line(attention_mask_shape_file).strip().split()))
        attention_mask = np.fromfile(
            forward_dir / 'attention_mask.npy',
            dtype=np.uint64,
        ).reshape(shape)

    compare_values(forward_dir, attention_mask)
    compare_gradients(gradient_dir)
