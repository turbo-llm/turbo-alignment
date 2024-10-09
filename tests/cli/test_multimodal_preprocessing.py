from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.constants import FIXTURES_PATH
from turbo_alignment.cli import app

runner = CliRunner()


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
