from pydantic import Field

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


class VLLMSampleingSettings(ExtraFieldsNotAllowedBaseModel):
    n: int = 1
    repetition_penalty: float = 1.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    max_tokens: int = 15
    logprobs: int | None = None


class VLLMGeneratorSettings(ExtraFieldsNotAllowedBaseModel):
    sampling_params: VLLMSampleingSettings = Field(default_factory=VLLMSampleingSettings)
    best_of: int = 1
    use_beam_search: bool = False
    stop_strings: str | list[str] = '</s>'
