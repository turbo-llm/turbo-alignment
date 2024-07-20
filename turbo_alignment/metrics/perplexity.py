import torch
from transformers import PreTrainedTokenizerBase

from turbo_alignment.metrics.metric import Metric
from turbo_alignment.metrics.utils import calculate_cross_entropy
from turbo_alignment.settings.metric import ElementWiseScores, MetricResults, MetricType


@Metric.register(MetricType.PERPLEXITY)
class PerplexityMetric(Metric):
    def compute(self, **kwargs) -> list[MetricResults]:
        tokenizer: PreTrainedTokenizerBase = kwargs.get('tokenizer', None)
        logits: torch.Tensor = kwargs.get('logits', None)
        labels: torch.Tensor = kwargs.get('answer_tokens_ids', None)
        dataset_name: str = kwargs.get('dataset_name', '')

        if logits is None:
            raise ValueError('logits should not be None')

        if labels is None:
            raise ValueError('labels should not be None')

        if tokenizer is None:
            raise ValueError('tokenizer should not be None')

        perplexity_values = []
        for logit, answer in zip(logits, labels):
            perplexity_values.extend(self.calculate_perplexity(logit, answer, tokenizer.pad_token_id))

        return [
            MetricResults(
                element_wise_scores=[
                    ElementWiseScores(
                        label=dataset_name + '@@' + 'perplexity',
                        values=perplexity_values,
                    )
                ],
                need_average=need_average,
            )
            for need_average in self._settings.need_average
        ]

    @staticmethod
    def calculate_perplexity(logits: torch.Tensor, labels: torch.Tensor, pad_token_id: int) -> list[float]:
        ppl = calculate_cross_entropy(logits, labels, pad_token_id, reduction='none')
        return torch.exp(ppl).detach().cpu().tolist()
