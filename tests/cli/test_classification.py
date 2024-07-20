from pathlib import Path

from typer.testing import CliRunner

from tests.constants import FIXTURES_PATH
from turbo_alignment.cli import app

runner = CliRunner()


def test_classification_train():
    result = runner.invoke(
        app,
        [
            'train_classification',
            '--experiment_settings_path',
            FIXTURES_PATH / 'configs/train/classification/base.json',
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert Path('test_train_classification_output').is_dir()


if __name__ == '__main__':
    test_classification_train()
