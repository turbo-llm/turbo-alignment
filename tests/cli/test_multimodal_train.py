from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.constants import FIXTURES_PATH
from turbo_alignment.cli import app
from turbo_alignment.settings.pipelines.train.multimodal import (
    MultimodalTrainExperimentSettings,
)

runner = CliRunner()


@pytest.mark.skip()
@pytest.mark.parametrize(
    'config_path',
    [
        FIXTURES_PATH / 'configs/train/multimodal/llama_llava_base_clip.json',
    ],
)
def test_multimodal_train_mlp_without_preprocessing(config_path: Path):
    result = runner.invoke(
        app, ['train_multimodal', '--experiment_settings_path', str(config_path)], catch_exceptions=False
    )
    assert result.exit_code == 0
    assert MultimodalTrainExperimentSettings.parse_file(config_path).log_path.is_dir()


@pytest.mark.skip()
@pytest.mark.parametrize(
    'config_path',
    [
        FIXTURES_PATH / 'configs/utils/preprocess/images.json',
    ],
)
def test_multimodal_preprocessing(config_path: Path):
    result = runner.invoke(
        app, ['preprocess_multimodal_dataset', '--settings_path', str(config_path)], catch_exceptions=False
    )
    assert result.exit_code == 0


@pytest.mark.skip()
@pytest.mark.parametrize(
    'config_path',
    [
        FIXTURES_PATH / 'configs/train/multimodal/llama_c_abs_clip_pickle.json',
    ],
)
def test_multimodal_train_c_abs_with_preprocessing(config_path: Path):
    result = runner.invoke(
        app, ['train_multimodal', '--experiment_settings_path', str(config_path)], catch_exceptions=False
    )
    assert result.exit_code == 0
    assert MultimodalTrainExperimentSettings.parse_file(config_path).log_path.is_dir()
