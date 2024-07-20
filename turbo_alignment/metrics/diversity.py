from collections import defaultdict
from typing import Any

import numpy as np
from scipy.stats import entropy
from transformers import PreTrainedTokenizerBase

from turbo_alignment.metrics.metric import Metric
from turbo_alignment.settings.metric import ElementWiseScores, MetricResults, MetricType


@Metric.register(MetricType.DIVERSITY)
class DiversityMetric(Metric):
    def compute(self, **kwargs) -> list[MetricResults]:
        tokenizer: PreTrainedTokenizerBase = kwargs.get('tokenizer', None)
        predictions: list[list[str]] = kwargs.get('predictions', None)
        dataset_name: str = kwargs.get('dataset_name', '')

        if predictions is None:
            raise ValueError('predictions should not be None')

        if tokenizer is None:
            raise ValueError('tokenizer should not be None')

        element_wise_diversity_scores = [
            ElementWiseScores(
                label=dataset_name + '@@' + 'diversity',
                values=[self.average_token_entropy(answer_group, tokenizer) for answer_group in predictions],
            )
        ]

        return [
            MetricResults(element_wise_scores=element_wise_diversity_scores, need_average=need_average)
            for need_average in self._settings.need_average
        ]

    def average_token_entropy(self, answer_group: list[str], tokenizer: PreTrainedTokenizerBase) -> float:
        entropies = [self.token_entropy(answer, tokenizer) for answer in answer_group]
        if entropies:
            return sum(entropies) / len(entropies)

        return np.nan

    @staticmethod
    def token_entropy(sample: str, tokenizer: PreTrainedTokenizerBase) -> float:
        stats: dict[int, Any] = defaultdict(int)
        num_tokens = 0
        tokens = tokenizer.encode(sample)
        for t in tokens:
            if t == tokenizer.pad_token_id:
                continue
            stats[t] += 1
            num_tokens += 1
        for k in stats.keys():
            stats[k] /= num_tokens

        return entropy(list(stats.values()))
