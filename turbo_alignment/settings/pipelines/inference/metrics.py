from pathlib import Path

from turbo_alignment.settings.base import ExtraFieldsNotAllowedBaseModel
from turbo_alignment.settings.metric import MetricSettings
from turbo_alignment.settings.pipelines.inference.base import (
    INFERENCE_DATASETS_SETTINGS,
)
from turbo_alignment.settings.pipelines.inference.chat import (
    ChatSingleModelInferenceSettings,
)


class MetricsSettings(ExtraFieldsNotAllowedBaseModel):
    inference_settings: ChatSingleModelInferenceSettings
    metric_settings: list[MetricSettings] = []

    dataset_settings: INFERENCE_DATASETS_SETTINGS
    save_path: Path
