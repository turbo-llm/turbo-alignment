import abc

from turbo_alignment.common.registry import Registrable
from turbo_alignment.metrics.registry import MetricSettings
from turbo_alignment.settings.metric import MetricResults


class Metric(abc.ABC, Registrable):
    def __init__(self, settings: MetricSettings) -> None:
        self._settings: MetricSettings = settings

    @abc.abstractmethod
    def compute(self, **kwargs) -> list[MetricResults]:
        ...
