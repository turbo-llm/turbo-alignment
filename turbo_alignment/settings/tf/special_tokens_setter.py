from turbo_alignment.settings.base import ExtraFieldsNotAllowedBaseModel


class SpecialTokensSettings(ExtraFieldsNotAllowedBaseModel):
    bos_token: str | None
    eos_token: str
    pad_token: str | None = None
    unk_token: str | None = None
    sep_token: str | None = None
