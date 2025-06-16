from turbo_alignment.settings.base import ExtraFieldsNotAllowedBaseModel


class BaseGeneratorSettings(ExtraFieldsNotAllowedBaseModel):
    repetition_penalty: float = 1.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    stop_strings: str | list[str] = '</s>'


class GeneratorTransformersSettings(BaseGeneratorSettings):
    num_beams: int = 1
    max_new_tokens: int = 15
    num_return_sequences: int = 1
    do_sample: bool = True


class VLLMGeneratorSettings(BaseGeneratorSettings):
    n: int = 1
    max_tokens: int = 15
    logprobs: int | None = None
    best_of: int = 1
    filter_token_ids: list[int] | None = None
