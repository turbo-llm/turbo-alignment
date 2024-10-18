from typing import Literal

from turbo_alignment.settings.base import ExtraFieldsNotAllowedBaseModel
from turbo_alignment.settings.datasets.base import (
    BaseDatasetSettings,
    DatasetType,
    MultiDatasetSettings,
)


class ChatPromptTemplate(ExtraFieldsNotAllowedBaseModel):
    prefix_template: str
    suffix_template: str
    role_tag_mapping: dict[str, str]


class ChatDatasetSettings(BaseDatasetSettings):
    dataset_type: Literal[DatasetType.CHAT] = DatasetType.CHAT

    # FIXME: this settings are strange for inference dataset
    only_last_replica_loss: bool = False
    only_answer_loss: bool = True
    random_cut: bool = False  # TODO: not common property for train/inference

    keep_end: bool | None = None
    max_tokens_count: int | None
    prompt_template: ChatPromptTemplate
    ignore_system_prompt: bool = False


class ChatMultiDatasetSettings(ChatDatasetSettings, MultiDatasetSettings):
    ...
