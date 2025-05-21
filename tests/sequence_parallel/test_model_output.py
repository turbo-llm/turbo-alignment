import json
import os
import pathlib
import subprocess
from typing import Any

import numpy as np
import pytest
import yaml  # type: ignore[import-untyped]

from tests.constants import FIXTURES_PATH
from tests.sequence_parallel.consts import GEMMA_MODEL_PATH, QWEN_MODEL_PATH, QWEN3_MODEL_PATH
from tests.sequence_parallel.utils import read_first_line
from tests.sequence_parallel.compare_gradients import compare as compare_gradients
from tests.sequence_parallel.compare_values import compare as compare_values
from tests.sequence_parallel.marks import has_gemma_model, has_two_gpus, has_qwen_model

DPO_SETTINGS_PATH = FIXTURES_PATH / 'configs' / 'train' / 'dpo' / 'dpo_with_seq_p.json'
SFT_SETTINGS_PATH = FIXTURES_PATH / 'configs' / 'train' / 'sft' / 'base_with_seq_p.json'
SFT_PEFT_SETTINGS_PATH = FIXTURES_PATH / 'configs' / 'train' / 'sft' / 'peft_with_seq_p.json'
RM_SETTINGS_PATH = FIXTURES_PATH / 'configs' / 'train' / 'rm' / 'base_with_seq_p.json'

DIR = pathlib.Path(__file__).parent


def patch_settings(
    settings: dict[str, Any],
    checkpoint_dir: pathlib.Path,
    model_path: str,
    model_type: str,
) -> dict[str, Any]:
    settings = settings.copy()
    settings['model_settings']['model_path'] = model_path
    settings['model_settings']['model_type'] = model_type
    settings['log_path'] = str(checkpoint_dir.absolute())
    return settings


@pytest.mark.parametrize(
    'task_type,settings_path',
    [
        pytest.param('dpo', DPO_SETTINGS_PATH, id='dpo'),
        pytest.param('sft', SFT_SETTINGS_PATH, id='sft'),
        pytest.param('sft', SFT_PEFT_SETTINGS_PATH, id='sft_lora', marks=pytest.mark.xfail),
        pytest.param('rm', RM_SETTINGS_PATH, id='rm')
    ],
)
@pytest.mark.skipif(not has_two_gpus(), reason='At least two gpu are required')
@pytest.mark.parametrize(
    'model_path,model_type',
    [
        pytest.param(
            GEMMA_MODEL_PATH,
            'gemma_with_ulysses',
            id='gemma',
            marks=pytest.mark.skipif(not has_gemma_model(), reason='Gemma model not found'),
        ),
        pytest.param(
            QWEN_MODEL_PATH,
            'qwen_with_ulysses',
            id='qwen',
            marks=pytest.mark.skipif(not has_qwen_model(), reason='Qwen model not found'),
        ),
        pytest.param(
            QWEN3_MODEL_PATH,
            'qwen3_with_ulysses',
            id='qwen3',
            marks=pytest.mark.skipif(not has_qwen_model(), reason='Qwen3 model not found'),
        ),
    ],
)
@pytest.mark.parametrize(
    'launch_mode',
    # ['deepspeed', 'accelerate'],
    ['deepspeed'],
)
def test_model_output(
    task_type: str,
    settings_path: pathlib.Path,
    model_path: str,
    model_type: str,
    launch_mode: str,
    tmp_path_factory: pytest.TempPathFactory,
):
    if task_type == 'rm' and 'qwen3' not in model_type:
        pytest.skip(reason='Only Qwen3 supported for rm')

    if task_type == 'rm':
        model_type = 'seq_cls_' + model_type

    env = os.environ.copy()

    script_path = str((DIR / 'test_dpo.py').absolute())

    gradient_dir = tmp_path_factory.mktemp('gradient').absolute()
    forward_dir = tmp_path_factory.mktemp('forward').absolute()
    checkpoint_dir = tmp_path_factory.mktemp('checkpoint').absolute()
    settings_dir = tmp_path_factory.mktemp('settings').absolute()
    new_settings_path = settings_dir / 'settings.json'

    with settings_path.open('r', encoding='utf-8') as input_:
        settings = json.load(input_)

    new_settings = patch_settings(settings, checkpoint_dir, model_path, model_type)

    with new_settings_path.open('w', encoding='utf-8') as output:
        json.dump(new_settings, output, indent=4, ensure_ascii=False)

    print(new_settings)

    env['GRADIENT_HOOK_OUTPUT_DIR'] = str(gradient_dir)
    env['FORWARD_HOOK_OUTPUT_DIR'] = str(forward_dir)
    env['CREATE_GRADIENT_HOOK'] = '1'

    if launch_mode == 'accelerate':
        src_accelerate_config_path = pathlib.Path(__file__).parent / 'accelerate_config.yml'
        with src_accelerate_config_path.open('r', encoding='utf-8') as input_:
            accelerate_config = yaml.safe_load(input_)

        deepspeed_cfg = new_settings['trainer_settings']['deepspeed']
        config_dir = tmp_path_factory.mktemp('configs')
        deepspeed_cfg_path = config_dir / 'deepspeed.json'
        with deepspeed_cfg_path.open('w', encoding='utf-8') as output:
            json.dump(deepspeed_cfg, output, indent=4)

        accelerate_config['deepspeed_config']['deepspeed_config_file'] = str(deepspeed_cfg_path.absolute())
        new_accelerate_config_path = config_dir / 'accelerate.yaml'
        with new_accelerate_config_path.open('w', encoding='utf-8') as output:
            yaml.safe_dump(accelerate_config, output)

        common_cmd_line = [
            'accelerate',
            'launch',
            '--config_file',
            str(new_accelerate_config_path.absolute()),
            '--machine_rank',
            '0',
            '--num_machines',
            '1',
        ]

        def build_middle_args(num_gpus: int):
            return ['--num_processes', str(num_gpus)]

    elif launch_mode == 'deepspeed':
        common_cmd_line = [
            'deepspeed',
            '--no_local_rank',
        ]

        def build_middle_args(num_gpus: int):
            return ['--num_gpus', str(num_gpus)]

    else:
        raise ValueError(f'Unsupported {launch_mode=}')

    script_args = [
        script_path,
        '--task-type',
        task_type,
        '--settings-path',
        str(new_settings_path.absolute()),
    ]

    print('Run with ulessys')
    # subprocess.check_call(common_cmd_line + ['--num_gpus', '2'] + script_args, env=env)
    subprocess.check_call(common_cmd_line + build_middle_args(2) + script_args, env=env)
    print('Run vanilla')

    subprocess.check_call(common_cmd_line + build_middle_args(1) + script_args + ['--make-model-vanilla'], env=env)

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
