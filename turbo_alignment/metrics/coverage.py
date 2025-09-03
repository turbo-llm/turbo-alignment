import numpy as np

from turbo_alignment.metrics.metric import Metric
from turbo_alignment.metrics.registry import CoverageSettings
from turbo_alignment.settings.metric import ElementWiseScores, MetricResults, MetricType


def check_substring_not_in_sting(prediction, target_substring):
    return 1 if target_substring.strip() not in prediction.strip() else 0


@Metric.register(MetricType.COVERAGE)
class CoverageMetric(Metric):

    def __init__(self, settings: CoverageSettings) -> None:
        super().__init__(settings=settings)
        self._settings: CoverageSettings = settings
        self.target_str = settings.target_string

    def compute(self, **kwargs) -> list[MetricResults]:
        references: list[list[str]] = kwargs.get('references', None)
        predictions: list[list[str]] = kwargs.get('predictions', None)
        dataset_name: str = kwargs.get('dataset_name', '')

        if references is None:
            raise ValueError('references should not be None')

        if predictions is None:
            raise ValueError('predictions should not be None')

        return [
            MetricResults(
                element_wise_scores=[
                    ElementWiseScores(
                        label=dataset_name + '@@' + 'coverage',
                        values=[
                            check_substring_not_in_sting(prediction, target_substring=self.target_str)
                            for reference_list, prediction_list in zip(references, predictions)
                            for reference, prediction in zip(reference_list, prediction_list)
                        ],
                    )
                ],
                need_average=need_average,
            )
            for need_average in self._settings.need_average
        ]
