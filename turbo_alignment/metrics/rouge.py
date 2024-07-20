from collections import defaultdict

from nltk.tokenize import wordpunct_tokenize
from rouge_score import rouge_scorer
from rouge_score.tokenizers import Tokenizer

from turbo_alignment.metrics.metric import Metric
from turbo_alignment.metrics.registry import RougeSettings
from turbo_alignment.settings.metric import ElementWiseScores, MetricResults, MetricType


class CustomTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        return wordpunct_tokenize(text)


@Metric.register(MetricType.ROUGE)
class RougeMetric(Metric):
    def __init__(self, settings: RougeSettings) -> None:
        super().__init__(settings=settings)
        self._settings: RougeSettings = settings
        rouge_types = []

        if settings.need_compute_rouge_l:
            rouge_types.append('rougeL')

        for n in settings.need_compute_rouge_n:
            rouge_types.append(f'rouge{n}')

        self._rouge_scorer = rouge_scorer.RougeScorer(rouge_types=rouge_types, tokenizer=CustomTokenizer())

    def compute(self, **kwargs) -> list[MetricResults]:
        references: list[str] = kwargs.get('references', None)
        predictions: list[str] = kwargs.get('predictions', None)

        element_wise_scores = self._compute_element_wise(references=references, predictions=predictions)

        return [
            MetricResults(element_wise_scores=element_wise_scores, need_average=need_average)
            for need_average in self._settings.need_average
        ]

    def _compute_element_wise(self, references: list[str], predictions: list[str]) -> list[ElementWiseScores]:
        element_wise_scores = []
        metric_label_to_element_wise_scores = defaultdict(list)
        for reference_batch, prediction_batch in zip(references, predictions):
            for reference, prediction in zip(reference_batch, prediction_batch):
                scores = self._rouge_scorer.score(
                    target=reference,
                    prediction=prediction,
                )

                for label, value in scores.items():
                    metric_label_to_element_wise_scores[f'{label}_precision'].append(value.precision)
                    metric_label_to_element_wise_scores[f'{label}_recall'].append(value.recall)
                    metric_label_to_element_wise_scores[f'{label}_fmeasure'].append(value.fmeasure)

        for label, values in metric_label_to_element_wise_scores.items():
            element_wise_scores.append(ElementWiseScores(label=label, values=values))

        return element_wise_scores
