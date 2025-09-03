from collections import defaultdict

from turbo_alignment.metrics.metric import Metric
from turbo_alignment.metrics.registry import StringAccuracySettings
from turbo_alignment.settings.metric import ElementWiseScores, MetricResults, MetricType


def accuracy_score(target, prediction):
    return 1 if target.strip() in prediction.strip() else 0


@Metric.register(MetricType.STR_ACCURACY)
class StringAccuracyMetric(Metric):

    def __init__(self, settings: StringAccuracySettings) -> None:
        super().__init__(settings=settings)
        self._settings: StringAccuracySettings = settings

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
                        label=dataset_name + '@@' + 'accuracy',
                        values=[
                            accuracy_score(reference, prediction)
                            for reference_list, prediction_list in zip(references, predictions)
                            for reference, prediction in zip(reference_list, prediction_list)
                        ],
                    )
                ],
                need_average=need_average,
            )
            for need_average in self._settings.need_average
        ]

    # def __init__(self, settings: StringAccuracySettings) -> None:
    #     super().__init__(settings=settings)
    #     self._settings: StringAccuracySettings = settings
    #
    # def compute(self, **kwargs) -> list[MetricResults]:
    #     references: list[str] = kwargs.get('references', None)
    #     predictions: list[str] = kwargs.get('predictions', None)
    #
    #     element_wise_scores = self._compute_element_wise(references=references, predictions=predictions)
    #
    #     return [
    #         MetricResults(element_wise_scores=element_wise_scores, need_average=need_average)
    #         for need_average in self._settings.need_average
    #     ]
    #
    # def _compute_element_wise(self, references: list[str], predictions: list[str]) -> list[ElementWiseScores]:
    #     element_wise_scores = []
    #     metric_label_to_element_wise_scores = defaultdict(list)
    #     for reference_batch, prediction_batch in zip(references, predictions):
    #         for reference, prediction in zip(reference_batch, prediction_batch):
    #             scores = accuracy_score(
    #                 target=reference,
    #                 prediction=prediction,
    #             )
    #
    #             metric_label_to_element_wise_scores['accuracy'].append(scores)
    #
    #     for label, values in metric_label_to_element_wise_scores.items():
    #         element_wise_scores.append(ElementWiseScores(label=label, values=values))
    #
    #     return element_wise_scores
