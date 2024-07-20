from pathlib import Path

from typer.testing import CliRunner

from tests.constants import FIXTURES_PATH
from turbo_alignment.cli import app

runner = CliRunner()


def test_convert_to_base():
    result = runner.invoke(
        app,
        ['convert_to_base', '--settings_path', FIXTURES_PATH / 'configs/utils/convert_to_base/merge_debug.json'],
    )
    assert result.exit_code == 0
    assert Path('test_merge_output').is_dir()
