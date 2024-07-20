from turbo_alignment.settings.base import ExtraFieldsNotAllowedBaseModel


class CustomChatGenerationSettings(ExtraFieldsNotAllowedBaseModel):
    skip_special_tokens: bool = True
    generation_eos_token: str = '</s>'
    remove_prompt: bool = True
    batch: int = 1
