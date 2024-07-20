from typer.testing import CliRunner

from tests.constants import FIXTURES_PATH
from turbo_alignment.cli import app

runner = CliRunner()


def test_rso_sampling():
    result = runner.invoke(
        app,
        ['rso_sample', '--experiment_settings_path', FIXTURES_PATH / 'configs/sampling/rso.json'],
    )
    assert result.exit_code == 0


def test_random_sampling():
    result = runner.invoke(
        app,
        ['random_sample', '--experiment_settings_path', FIXTURES_PATH / 'configs/sampling/base.json'],
    )
    assert result.exit_code == 0


def test_rm_sampling():
    result = runner.invoke(
        app,
        ['rm_sample', '--experiment_settings_path', FIXTURES_PATH / 'configs/sampling/rm.json'],
    )
    assert result.exit_code == 0
