from pathlib import Path

from turbo_alignment.settings.base import ExtraFieldsNotAllowedBaseModel
from turbo_alignment.settings.model import (
    ModelForPeftSettings,
    PreTrainedAdaptersModelSettings,
    PreTrainedModelSettings,
)


class RAGIndexSettings(ExtraFieldsNotAllowedBaseModel):
    index_path: Path
    passages_path: Path


class RAGRetrievalSettings(ExtraFieldsNotAllowedBaseModel):
    n_docs: int = 1
    title_sep: str | None = ' / '
    doc_sep: str | None = ' // '
    prefix: str | None = ''  # 'query:' for e5
    max_doc_length: int = 300
    retrieval_vector_size: int = 768
    retrieval_batch_size: int = 8
    query_encoder_max_length: int = 512
    encoder_learning_rate: float | None = None


class RAGPreTrainedModelSettings(ExtraFieldsNotAllowedBaseModel):
    generator_settings: ModelForPeftSettings | PreTrainedAdaptersModelSettings
    question_encoder_settings: PreTrainedModelSettings
    index_settings: RAGIndexSettings
    retrieval_settings: RAGRetrievalSettings
