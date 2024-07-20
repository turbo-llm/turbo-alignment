import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from turbo_alignment.metrics.metric import Metric
from turbo_alignment.metrics.registry import SelfBleuSettings
from turbo_alignment.settings.metric import ElementWiseScores, MetricResults, MetricType


@Metric.register(MetricType.SELF_BLEU)
class SelfBleuMetric(Metric):
    def __init__(self, settings: SelfBleuSettings) -> None:
        super().__init__(settings=settings)
        self._settings: SelfBleuSettings = settings

    def compute(self, **kwargs) -> list[MetricResults]:
        predictions: list[list[str]] = kwargs.get('predictions', None)
        dataset_name: str = kwargs.get('dataset_name', '')

        if predictions is None:
            raise ValueError('predictions should not be None')

        return [
            MetricResults(
                element_wise_scores=[
                    ElementWiseScores(
                        label=dataset_name + '@@' + 'self_bleu',
                        values=[self.self_bleu(answers) for answers in predictions],
                    )
                ],
                need_average=need_average,
            )
            for need_average in self._settings.need_average
        ]

    def self_bleu(self, answers: list[str]) -> float:
        weight = tuple((1.0 / self._settings.ngram for _ in range(self._settings.ngram)))
        result = []
        sentence_num = len(answers)
        for index in range(sentence_num):
            hypothesis = answers[index]
            other = answers[:index] + answers[index + 1 :]
            result.append(self._calc_bleu(other, hypothesis, weight))
        return np.mean(result).item()

    def _calc_bleu(self, reference: list[str], hypothesis: str, weight: tuple[float, ...]) -> list[float]:
        return sentence_bleu(reference, hypothesis, weight, smoothing_function=SmoothingFunction().method1)
