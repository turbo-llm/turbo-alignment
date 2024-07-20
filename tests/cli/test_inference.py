from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.constants import FIXTURES_PATH
from turbo_alignment.cli import app
from turbo_alignment.common.data.io import read_json, read_jsonl
from turbo_alignment.settings.pipelines.inference import (
    ChatInferenceExperimentSettings,
    InferenceExperimentSettings,
    RAGInferenceExperimentSettings,
)

runner = CliRunner()


@pytest.mark.parametrize(
    'config_path',
    [
        FIXTURES_PATH / 'configs/inference/sft/base.json',
    ],
)
def test_inference_chat(config_path: Path):
    result = runner.invoke(
        app,
        ['inference_chat', '--inference_settings_path', str(config_path)],
    )
    assert result.exit_code == 0
    inference_settings = ChatInferenceExperimentSettings.parse_file(config_path)
    info_save_file = inference_settings.save_path / Path('info.json')
    assert info_save_file.is_file()
    info_file = read_json(info_save_file)
    assert len(info_file) != 0
    for filename in info_file:
        filepath = inference_settings.save_path / filename
        assert filepath.is_file()
        assert len(read_jsonl(filepath)) != 0


@pytest.mark.parametrize(
    'config_path',
    [
        FIXTURES_PATH / 'configs/inference/classification/base.json',
    ],
)
def test_inference_classification(config_path: Path):
    result = runner.invoke(
        app,
        ['inference_classification', '--inference_settings_path', str(config_path)],
    )
    assert result.exit_code == 0
    inference_settings = InferenceExperimentSettings.parse_file(config_path)
    info_save_file = inference_settings.save_path / Path('info.json')
    assert info_save_file.is_file()
    info_file = read_json(info_save_file)
    assert len(info_file) != 0
    for filename in info_file:
        filepath = inference_settings.save_path / filename
        assert filepath.is_file()
        assert len(read_jsonl(filepath)) != 0


@pytest.mark.parametrize(
    'config_path',
    [
        FIXTURES_PATH / 'configs/inference/rag/base.json',
    ],
)
def test_inference_rag(config_path: Path):
    result = runner.invoke(
        app,
        ['inference_rag', '--inference_settings_path', str(config_path)],
    )
    assert result.exit_code == 0
    inference_settings = RAGInferenceExperimentSettings.parse_file(config_path)
    info_save_file = inference_settings.save_path / Path('info.json')
    assert info_save_file.is_file()
    info_file = read_json(info_save_file)
    assert len(info_file) != 0
    for filename in info_file:
        filepath = inference_settings.save_path / filename
        assert filepath.is_file()
        assert len(read_jsonl(filepath)) != 0
