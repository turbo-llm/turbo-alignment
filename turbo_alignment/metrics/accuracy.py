from typing import Optional

from transformers import PreTrainedTokenizerBase

from turbo_alignment.metrics.metric import Metric
from turbo_alignment.settings.metric import ElementWiseScores, MetricResults, MetricType


@Metric.register(MetricType.ACCURACY)
class AccuracyMetric(Metric):
    def compute(self, **kwargs) -> list[MetricResults]:
        tokenizer: Optional[PreTrainedTokenizerBase] = kwargs.get('tokenizer', None)  # type: ignore[assignment]
        references: Optional[list[list[str]]] = kwargs.get('references', None)  # type: ignore[assignment]
        predictions: Optional[list[list[str]]] = kwargs.get('predictions', None)  # type: ignore[assignment]
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
                            int(reference == self._remove_suffix(prediction, tokenizer))
                            for reference_list, prediction_list in zip(references, predictions)
                            for reference, prediction in zip(reference_list, prediction_list)
                        ],
                    )
                ],
                need_average=need_average,
            )
            for need_average in self._settings.need_average
        ]

    @staticmethod
    def _remove_suffix(prediction: str, tokenizer: PreTrainedTokenizerBase) -> str:
        return prediction.removesuffix(tokenizer.pad_token).removesuffix(tokenizer.eos_token)
