from abc import abstractmethod
from typing import Generic, TypeVar

from turbo_alignment.common.data.io import write_jsonl
from turbo_alignment.dataset.sampling.models import SamplingDatasetRecord
from turbo_alignment.pipelines.base import BaseStrategy
from turbo_alignment.settings.pipelines.sampling import BaseSamplingSettings

SamplingSettingsT = TypeVar('SamplingSettingsT', bound=BaseSamplingSettings)


class BaseSamplingStrategy(BaseStrategy, Generic[SamplingSettingsT]):
    @abstractmethod
    def sample(self, experiment_settings: SamplingSettingsT) -> list[SamplingDatasetRecord]:
        ...

    def run(self, experiment_settings: SamplingSettingsT) -> None:
        experiment_settings.save_path.mkdir(parents=True, exist_ok=True)

        sampled_outputs: list[SamplingDatasetRecord] = self.sample(experiment_settings)

        write_jsonl([out.dict() for out in sampled_outputs], experiment_settings.save_path / 'sampled.jsonl')
