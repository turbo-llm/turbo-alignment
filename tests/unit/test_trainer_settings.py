import json

import pytest

from tests.constants import FIXTURES_PATH
from turbo_alignment.settings import pipelines as pipeline_settings


@pytest.mark.parametrize(
    'config_path, settings_cls',
    [
        (FIXTURES_PATH / 'configs/train/sft/base.json', pipeline_settings.SftTrainExperimentSettings),
        (FIXTURES_PATH / 'configs/train/dpo/base.json', pipeline_settings.DPOTrainExperimentSettings),
        (FIXTURES_PATH / 'configs/train/kto/base.json', pipeline_settings.KTOTrainExperimentSettings),
    ],
)
def test_evaluation_strategy_deprecation(config_path, settings_cls):
    match_str = (
        "'evaluation_strategy' is deprecated and will be removed in a future version. Use 'eval_strategy' instead."
    )
    with pytest.warns(FutureWarning, match=match_str):
        with open(config_path) as f:
            json_data = json.load(f)
        json_data["trainer_settings"]["evaluation_strategy"] = json_data["trainer_settings"].pop("eval_strategy")
        settings = settings_cls.model_validate(json_data)

    trainer_settings = settings.trainer_settings
    assert not hasattr(trainer_settings, 'evaluation_strategy')
    assert hasattr(trainer_settings, 'eval_strategy')
    assert trainer_settings.eval_strategy == 'steps'
