from turbo_alignment.settings.base import ExtraFieldsNotAllowedBaseModel


class CustomChatGenerationSettings(ExtraFieldsNotAllowedBaseModel):
    skip_special_tokens: bool = True
    remove_prompt: bool = True
    only_answer_logits: bool = True
    batch: int = 1
