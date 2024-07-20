import torch
from transformers import PreTrainedTokenizerBase

from turbo_alignment.metrics.metric import Metric
from turbo_alignment.settings.metric import ElementWiseScores, MetricResults, MetricType


@Metric.register(MetricType.LENGTH)
class LengthMetric(Metric):
    def compute(self, **kwargs) -> list[MetricResults]:
        tokenizer: PreTrainedTokenizerBase = kwargs.get('tokenizer', None)
        answer_tokens_ids: list[torch.Tensor] = kwargs.get('answer_tokens_ids', None)
        dataset_name: str = kwargs.get('dataset_name', '')

        if answer_tokens_ids is None:
            raise ValueError('answer_tokens_ids should not be None')

        if tokenizer is None:
            raise ValueError('tokenizer should not be None')

        return [
            MetricResults(
                element_wise_scores=[
                    ElementWiseScores(
                        label=dataset_name + '@@' + 'length',
                        values=[
                            (answer_tokens != tokenizer.pad_token_id).sum().item()
                            for answer_tokens in answer_tokens_ids
                        ],
                    )
                ],
                need_average=need_average,
            )
            for need_average in self._settings.need_average
        ]
