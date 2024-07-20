from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.constants import FIXTURES_PATH
from turbo_alignment.cli import app
from turbo_alignment.settings.pipelines.train.rm import RMTrainExperimentSettings

runner = CliRunner()


@pytest.mark.parametrize(
    'config_path',
    [
        FIXTURES_PATH / 'configs/train/rm/base.json',
    ],
)
def test_rm_train(config_path: Path):
    result = runner.invoke(
        app,
        ['train_rm', '--experiment_settings_path', str(config_path)],
    )
    assert result.exit_code == 0
    assert RMTrainExperimentSettings.parse_file(config_path).log_path.is_dir()
