from collections import defaultdict

from turbo_alignment.metrics.metric import Metric
from turbo_alignment.settings.metric import ElementWiseScores, MetricResults, MetricType


@Metric.register(MetricType.DIST_N)
class DistinctnessMetric(Metric):
    def compute(self, **kwargs) -> list[MetricResults]:
        predictions: list[list[str]] = kwargs.get('predictions', None)
        dataset_name: str = kwargs.get('dataset_name', '')

        if predictions is None:
            raise ValueError('predictions should not be None')

        dist_n = defaultdict(list)
        for prompt_answers in predictions:
            ans_dist_n = self.distinctness(prompt_answers)
            for label, value in ans_dist_n.items():
                dist_n[label].append(value)

        return [
            MetricResults(
                element_wise_scores=[
                    ElementWiseScores(label=dataset_name + '@@' + label, values=value)
                    for label, value in dist_n.items()
                ],
                need_average=need_average,
            )
            for need_average in self._settings.need_average
        ]

    @staticmethod
    def distinctness(answers: list[str]) -> dict[str, float]:
        unigrams, bigrams, trigrams = set(), set(), set()
        total_words = 0

        for answer in answers:
            words = answer.split(' ')
            total_words += len(words)
            unigrams.update(words)
            for i in range(len(words) - 1):
                bigrams.add(words[i] + '_' + words[i + 1])
            for i in range(len(words) - 2):
                trigrams.add(words[i] + '_' + words[i + 1] + '_' + words[i + 2])

        return {
            'dist_1': len(unigrams) / total_words,
            'dist_2': len(bigrams) / total_words,
            'dist_3': len(trigrams) / total_words,
        }
