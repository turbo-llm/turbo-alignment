# from pathlib import Path

# import pytest
# from typer.testing import CliRunner

# from tests.constants import FIXTURES_PATH
# from turbo_alignment.cli import app
# from turbo_alignment.settings.pipelines.train.dpo import DPOTrainExperimentSettings

# runner = CliRunner()


# @pytest.mark.parametrize(
#     'config_path',
#     [
#         FIXTURES_PATH / 'configs/train/dpo/base.json',
#         FIXTURES_PATH / 'configs/train/dpo/simpo.json',
#     ],
# )
# def test_dpo_train(config_path: Path):
#     result = runner.invoke(
#         app,
#         ['train_dpo', '--experiment_settings_path', str(config_path)],
#     )
#     assert result.exit_code == 0
#     assert DPOTrainExperimentSettings.parse_file(config_path).log_path.is_dir()
