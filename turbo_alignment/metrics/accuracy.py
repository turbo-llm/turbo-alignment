from transformers import PreTrainedTokenizerBase

from turbo_alignment.metrics.metric import Metric
from turbo_alignment.settings.metric import ElementWiseScores, MetricResults, MetricType


@Metric.register(MetricType.ACCURACY)
class AccuracyMetric(Metric):
    def compute(self, **kwargs) -> list[MetricResults]:
        tokenizer: PreTrainedTokenizerBase = kwargs.get('tokenizer', None)
        references: list[list[str]] = kwargs.get('references', None)
        predictions: list[list[str]] = kwargs.get('predictions', None)
        dataset_name: str = kwargs.get('dataset_name', '')

        if references is None:
            raise ValueError('references should not be None')
        
        if predictions is None:
            raise ValueError('predictions should not be None')

        if tokenizer is None:
            raise ValueError('tokenizer should not be None')

        return [
            MetricResults(
                element_wise_scores=[
                    ElementWiseScores(
                        label=dataset_name + '@@' + 'accuracy',
                        values=[
                            int(reference[0] == prediction[0].removesuffix(tokenizer.eos_token))
                            for reference, prediction in zip(references, predictions)
                        ],
                    )
                ],
                need_average=need_average,
            )
            for need_average in self._settings.need_average
        ]
