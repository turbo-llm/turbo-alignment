import pytest

from turbo_alignment.metrics.accuracy import AccuracyMetric
from turbo_alignment.metrics.registry import AccuracySettings


class MockedTokenizer:
    def __init__(self, eos_token="</s>"):
        self.eos_token = eos_token

    def __call__(self, text):
        return text

@pytest.mark.parametrize("references,predictions,expected", [
    ([["Yes"], ["No"]], [["Yes</s>"], ["No</s>"]], [1, 1]),
    ([["Yes"], ["No"]], [["No</s>"], ["Yes</s>"]], [0, 0]),
    ([["Yes"], ["No"]], [["Yes</s>"], ["Yes</s>"]], [1, 0]),
])
def test_accuracy_metric(references, predictions, expected):
    settings = AccuracySettings(need_average=[False])
    metric = AccuracyMetric(settings=settings)
    tokenizer = MockedTokenizer()
    results = metric.compute(
        tokenizer=tokenizer,
        references=references,
        predictions=predictions,
    )
    scores = results[0].element_wise_scores[0].values
    assert scores == expected
