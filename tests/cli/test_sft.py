from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.constants import FIXTURES_PATH
from turbo_alignment.cli import app
from turbo_alignment.settings.pipelines.train.sft import SftTrainExperimentSettings

runner = CliRunner()


@pytest.mark.parametrize(
    'config_path',
    [
        FIXTURES_PATH / 'configs/train/sft/base.json',
    ],
)
def test_sft_train(config_path: Path):
    result = runner.invoke(app, ['train_sft', '--experiment_settings_path', str(config_path)], catch_exceptions=False)
    assert result.exit_code == 0
    assert SftTrainExperimentSettings.parse_file(config_path).log_path.is_dir()
