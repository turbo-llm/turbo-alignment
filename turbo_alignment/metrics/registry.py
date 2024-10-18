from enum import Enum

from turbo_alignment.common.registry import Registrable
from turbo_alignment.settings.base import ExtraFieldsNotAllowedBaseModel
from turbo_alignment.settings.datasets.chat import ChatPromptTemplate
from turbo_alignment.settings.metric import MetricType
from turbo_alignment.settings.model import (
    PreTrainedAdaptersModelSettings,
    PreTrainedModelSettings,
)
from turbo_alignment.settings.tf.tokenizer import TokenizerSettings


class KLType(str, Enum):
    SFT_MODEL: str = 'sft'
    REFERENCE_MODEL: str = 'reference'


class MetricSettingsRegistry(Registrable):
    ...


class MetricSettings(ExtraFieldsNotAllowedBaseModel):
    need_average: list[bool]


@MetricSettingsRegistry.register(MetricType.DIST_N)
class DistinctnessSettings(MetricSettings):
    ...


@MetricSettingsRegistry.register(MetricType.DIVERSITY)
class DiversitySettings(MetricSettings):
    ...


@MetricSettingsRegistry.register(MetricType.LENGTH)
class LengthSettings(MetricSettings):
    ...


@MetricSettingsRegistry.register(MetricType.KL)
class KLSettings(MetricSettings):
    ref_logits_type: KLType


@MetricSettingsRegistry.register(MetricType.METEOR)
class MeteorSettings(MetricSettings):
    compute_element_wise_meteor: bool = True
    element_wise_meteor_label: str = 'meteor'


@MetricSettingsRegistry.register(MetricType.PERPLEXITY)
class PerplexitySettings(MetricSettings):
    ...


@MetricSettingsRegistry.register(MetricType.REWARD)
class RewardSettings(MetricSettings):
    tokenizer_settings: TokenizerSettings
    model_settings: PreTrainedModelSettings | PreTrainedAdaptersModelSettings
    prompt_template: ChatPromptTemplate
    max_tokens_count: int
    system_prompt: str | None = None
    name: str = 'cherry_pick_dataset'
    micro_batch: int = 1


@MetricSettingsRegistry.register(MetricType.ROUGE)
class RougeSettings(MetricSettings):
    need_compute_rouge_n: list[int]
    need_compute_rouge_l: bool = True


@MetricSettingsRegistry.register(MetricType.SELF_BLEU)
class SelfBleuSettings(MetricSettings):
    ngram: int = 3


@MetricSettingsRegistry.register(MetricType.TOOL_CALL_METRICS)
class ToolMetricsSettings(MetricSettings):
    tool_activation_str: str


@MetricSettingsRegistry.register(MetricType.RETRIEVAL_UTILITY)
class RetrievalUtilitySettings(MetricSettings):
    doc_sep_symbol: str = '<doc_sep>'
