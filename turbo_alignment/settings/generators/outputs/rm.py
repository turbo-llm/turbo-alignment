from turbo_alignment.dataset.pair_preferences import PairPreferenceRecord
from turbo_alignment.dataset.sampling.models import SamplingDatasetRecord
from turbo_alignment.settings.generators.outputs.base import BaseInferenceOutput


class RMPairInferenceOutput(BaseInferenceOutput, PairPreferenceRecord):
    reward_w: float
    reward_l: float


class RMSamplingInferenceOutput(SamplingDatasetRecord):
    rewards: dict[str, float]
