from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.constants import FIXTURES_PATH
from turbo_alignment.cli import app
from turbo_alignment.settings.pipelines.train.rag import RAGTrainExperimentSettings

runner = CliRunner()


@pytest.mark.parametrize(
    'config_path',
    [
        FIXTURES_PATH / 'configs/train/rag/base.json',
    ],
)
def test_rag_train(config_path: Path):
    result = runner.invoke(
        app,
        ['train_rag', '--experiment_settings_path', str(config_path)],
    )
    assert result.exit_code == 0
    assert RAGTrainExperimentSettings.parse_file(config_path).log_path.is_dir()
