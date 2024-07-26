from pathlib import Path

from typer.testing import CliRunner

from tests.constants import FIXTURES_PATH
from turbo_alignment.cli import app

runner = CliRunner()


def test_merge_adapters_to_base():
    result = runner.invoke(
        app,
        [
            'merge_adapters_to_base',
            '--settings_path',
            FIXTURES_PATH / 'configs/utils/merge_adapters_to_base/merge_debug.json',
        ],
    )
    assert result.exit_code == 0
    assert Path('test_merge_output').is_dir()
