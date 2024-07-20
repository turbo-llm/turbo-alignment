import string

import evaluate
import nltk

from turbo_alignment.metrics.metric import Metric
from turbo_alignment.metrics.registry import MeteorSettings
from turbo_alignment.settings.metric import ElementWiseScores, MetricResults, MetricType


@Metric.register(MetricType.METEOR)
class MeteorMetric(Metric):
    def __init__(self, settings: MeteorSettings) -> None:
        super().__init__(settings=settings)
        self.compute_element_wise_meteor = settings.compute_element_wise_meteor
        self.element_wise_meteor_label = settings.element_wise_meteor_label
        self.meteor_metric = evaluate.load('meteor')

        nltk.download('wordnet')

    def compute(self, **kwargs) -> list[MetricResults]:
        references: list[str] = kwargs.get('references', None)
        predictions: list[str] = kwargs.get('predictions', None)
        dataset_name: str = kwargs.get('dataset_name', '')

        references_exist = references and len(references) > 0

        if not references_exist:
            return [MetricResults(element_wise_scores=[], need_average=self._settings.need_average)]

        element_wise_scores = []

        if self.compute_element_wise_meteor:
            element_wise_accuracy = self._compute_element_wise(
                references=references,
                predictions=predictions,
                label=dataset_name + '@@' + self.element_wise_meteor_label,
            )

            element_wise_scores.append(element_wise_accuracy)

        return [
            MetricResults(element_wise_scores=element_wise_scores, need_average=need_average)
            for need_average in self._settings.need_average
        ]

    def _compute_element_wise(self, references: list[str], predictions: list[str], label: str) -> ElementWiseScores:
        element_wise_meteor_scores = []
        for reference_batch, prediction_batch in zip(references, predictions):
            for reference, prediction in zip(reference_batch, prediction_batch):
                if len(reference) > 0 and len(prediction) > 0:
                    score = self._compute_meteor(reference=reference, prediction=prediction)
                else:
                    score = 0
                element_wise_meteor_scores.append(score)

        return ElementWiseScores(label=label, values=element_wise_meteor_scores)

    def _compute_meteor(self, reference: str, prediction: str) -> float:
        return self.meteor_metric.compute(
            predictions=[self._filter_punctuatuion(prediction)], references=[self._filter_punctuatuion(reference)]
        )['meteor']

    @staticmethod
    def _filter_punctuatuion(s: str) -> str:
        if s:
            return ''.join(char for char in s if char not in string.punctuation)
        return s
