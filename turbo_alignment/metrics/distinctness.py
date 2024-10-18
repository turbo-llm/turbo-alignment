from collections import defaultdict
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from turbo_alignment.metrics.metric import Metric
from turbo_alignment.metrics.registry import DistinctnessSettings
from turbo_alignment.settings.metric import ElementWiseScores, MetricResults, MetricType


@Metric.register(MetricType.DIST_N)
class DistinctnessMetric(Metric):
    def __init__(self, settings: DistinctnessSettings) -> None:
        super().__init__(settings=settings)
        self._settings: DistinctnessSettings = settings

    def compute(self, **kwargs) -> list[MetricResults]:
        predictions: list[list[str]] = kwargs.get('predictions', None)
        dataset_name: str = kwargs.get('dataset_name', '')
        tokenizer: PreTrainedTokenizerBase = kwargs.get("tokenizer", None)
        vocab_size: int = tokenizer.vocab_size

        if predictions is None:
            raise ValueError('predictions should not be None')

        dist_n = defaultdict(list)
        for prompt_answers in predictions:
            ans_dist_n = self.distinctness(prompt_answers, vocab_size, self._settings.ngram)
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
    def distinctness(answers: list[str], vocab_size: int, ngram: int) -> dict[str, float]:
        ngram_sets: list[set] = [set() for _ in range(ngram)]
        total_ngrams: list[int] = [0] * ngram

        for answer in answers:
            words = answer.split(' ')
            ngram_sets[0].update(words)
            total_ngrams[0] += len(words)

            for n in range(1, ngram):
                ngrams = ['_'.join(words[i : i + n + 1]) for i in range(len(words) - n)]
                ngram_sets[n].update(ngrams)
                total_ngrams[n] += len(ngrams)

        result = {}
        for n in range(ngram):
            result[f'dist_{n+1}'] = len(ngram_sets[n]) / total_ngrams[n] if total_ngrams[n] > 0 else 0
            try:
                result[f'ead_dist_{n+1}'] = (
                    len(ngram_sets[n]) / (vocab_size * (1 - ((vocab_size - 1) / vocab_size) ** total_ngrams[n]))
                    if total_ngrams[n] > 0
                    else 0
                )
            except ZeroDivisionError:
                result[f'ead_dist_{n+1}'] = 0

        result['dist_mean'] = sum(result[f'dist_{n+1}'] for n in range(ngram)) / ngram
        result['ead_dist_mean'] = sum(result[f'ead_dist_{n+1}'] for n in range(ngram)) / ngram
        return result
