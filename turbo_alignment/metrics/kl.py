import torch

from turbo_alignment.metrics.metric import Metric
from turbo_alignment.metrics.registry import KLSettings
from turbo_alignment.metrics.utils import logprobs_from_logits
from turbo_alignment.settings.metric import ElementWiseScores, MetricResults, MetricType


@Metric.register(MetricType.KL)
class KLMetric(Metric):
    def __init__(self, settings: KLSettings) -> None:
        super().__init__(settings=settings)
        self._settings: KLSettings = settings

    def compute(self, **kwargs) -> list[MetricResults]:
        answer_tokens_ids: list[torch.Tensor] = kwargs.get('answer_tokens_ids', None)
        logits: list[torch.Tensor] = kwargs.get('logits', None)
        metrics_kwargs = kwargs.get('metrics_kwargs', {})
        ref_logits: list[torch.Tensor] = metrics_kwargs.get(self._settings.ref_logits_type, None)
        dataset_name: str = kwargs.get('dataset_name', '')

        if answer_tokens_ids is None:
            raise ValueError('answer_tokens_ids should not be None')
        if logits is None:
            raise ValueError('logits should not be None')
        if ref_logits is None:
            raise ValueError(f'{self._settings.ref_logits_type} should not be None')

        ref_logprobs = self._logprobs(ref_logits, answer_tokens_ids)
        logprobs = self._logprobs(logits, answer_tokens_ids)

        element_wise_diversity_scores = [
            ElementWiseScores(
                label=dataset_name + '@@' + f'kl_with_{self._settings.ref_logits_type}',
                values=[(logprob - ref_logprob).mean().item() for logprob, ref_logprob in zip(logprobs, ref_logprobs)],
            )
        ]

        return [
            MetricResults(element_wise_scores=element_wise_diversity_scores, need_average=need_average)
            for need_average in self._settings.need_average
        ]

    @staticmethod
    def _logprobs(logits: list[torch.Tensor], labels: list[torch.Tensor]) -> list[torch.Tensor]:
        logprobs = []
        for item_logits, item_labels in zip(logits, labels):
            logprobs.append(logprobs_from_logits(logits=item_logits[:, :-1, :].cpu(), labels=item_labels[:, 1:].cpu()))
        return logprobs
