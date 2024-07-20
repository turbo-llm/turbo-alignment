from turbo_alignment.settings.base import ExtraFieldsNotAllowedBaseModel


class GeneratorTransformersSettings(ExtraFieldsNotAllowedBaseModel):
    # described in https://huggingface.co/docs/transformers/v4.42.0/en/main_classes/text_generation#transformers.GenerationConfig
    num_beams: int = 1
    max_new_tokens: int = 15
    repetition_penalty: float = 1.0
    num_return_sequences: int = 1
    do_sample: bool = True
    top_p: float = 1.0
    top_k: int = 50
    temperature: float = 1.0
