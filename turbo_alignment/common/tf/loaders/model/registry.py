from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)

from turbo_alignment.common.registry import Registrable
from turbo_alignment.settings.model import ModelType


class TransformersAutoModelRegistry(Registrable):
    ...


TransformersAutoModelRegistry.register(ModelType.CAUSAL)(AutoModelForCausalLM)
TransformersAutoModelRegistry.register(ModelType.SEQ_CLS)(AutoModelForSequenceClassification)
TransformersAutoModelRegistry.register(ModelType.ENC)(AutoModel)
