from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.constants import FIXTURES_PATH
from turbo_alignment.cli import app
from turbo_alignment.settings.pipelines.train.classification import (
    ClassificationTrainExperimentSettings,
)

runner = CliRunner()


@pytest.mark.parametrize(
    'config_path',
    [
        FIXTURES_PATH / 'configs/train/classification/base.json',
    ],
)
def test_classification_train(config_path: Path):
    result = runner.invoke(app, ['train_classification', '--experiment_settings_path', str(config_path)])
    assert result.exit_code == 0
    assert ClassificationTrainExperimentSettings.parse_file(config_path).log_path.is_dir()
