# from pathlib import Path

# import pytest
# from typer.testing import CliRunner

# from tests.constants import FIXTURES_PATH
# from turbo_alignment.cli import app
# from turbo_alignment.settings.pipelines.inference.multimodal import (
#     MultimodalInferenceExperimentSettings,
# )

# runner = CliRunner()


# @pytest.mark.parametrize(
#     'config_path',
#     [
#         FIXTURES_PATH / 'configs/inference/multimodal/llama_llava_clip_pickle.json',
#     ],
# )
# def test_multimodal_inference_mlp_with_preprocessing(config_path: Path):
#     result = runner.invoke(
#         app, ['inference_multimodal', '--inference_settings_path', str(config_path)], catch_exceptions=False
#     )
#     assert result.exit_code == 0
#     assert MultimodalInferenceExperimentSettings.parse_file(config_path).save_path.is_dir()
