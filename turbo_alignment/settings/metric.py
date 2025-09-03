from enum import Enum
from typing import Any, Callable

import numpy as np

from turbo_alignment.settings.base import ExtraFieldsNotAllowedBaseModel


class MetricType(str, Enum):
    ACCURACY = 'accuracy'
    STR_ACCURACY = 'str_accuracy'
    COVERAGE = 'coverage'
    ROUGE = 'rouge'
    METEOR = 'meteor'
    REWARD = 'reward'
    DIVERSITY = 'diversity'
    PERPLEXITY = 'perplexity'
    SELF_BLEU = 'self_bleu'
    LENGTH = 'length'
    DIST_N = 'dist_n'
    KL = 'kl'
    TOOL_CALL_METRICS = 'tool_call_metrics'
    RETRIEVAL_UTILITY = 'retrieval_utility'


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
