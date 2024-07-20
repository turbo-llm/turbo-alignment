from turbo_alignment.settings.base import ExtraFieldsNotAllowedBaseModel


class TokenizerSettings(ExtraFieldsNotAllowedBaseModel):
    use_fast: bool = False
    trust_remote_code: bool = False

    tokenizer_path: str | None = None

    tokenizer_kwargs: dict = {}
