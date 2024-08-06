from enum import Enum
from typing import Any, Callable

import numpy as np

from turbo_alignment.settings.base import ExtraFieldsNotAllowedBaseModel


class MetricType(str, Enum):
    ACCURACY: str = 'accuracy'
    ROUGE: str = 'rouge'
    METEOR: str = 'meteor'
    REWARD: str = 'reward'
    DIVERSITY: str = 'diversity'
    PERPLEXITY: str = 'perplexity'
    SELF_BLEU: str = 'self_bleu'
    LENGTH: str = 'length'
    DIST_N: str = 'dist_n'
    KL: str = 'kl'
    TOOL_CALL_METRICS: str = 'tool_call_metrics'
    RETRIEVAL_UTILITY: str = 'retrieval_utility'


class ElementWiseScores(ExtraFieldsNotAllowedBaseModel):
    label: str
    values: list[Any]
    average_function: Callable[[list[Any]], float] = np.mean


class MetricSettings(ExtraFieldsNotAllowedBaseModel):
    type: MetricType
    parameters: dict[str, Any]


class MetricResults(ExtraFieldsNotAllowedBaseModel):
    element_wise_scores: list[ElementWiseScores]
    need_average: bool = False
