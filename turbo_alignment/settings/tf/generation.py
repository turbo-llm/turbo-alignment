from turbo_alignment.settings.base import ExtraFieldsNotAllowedBaseModel


class GeneratorTransformersSettings(ExtraFieldsNotAllowedBaseModel):
    num_beams: int = 1
    max_new_tokens: int = 15
    repetition_penalty: float = 1.0
    num_return_sequences: int = 1
    do_sample: bool = True
    top_p: float = 1.0
    top_k: int = 50
    temperature: float = 1.0
    stop_strings: str | list[str] = '</s>'
