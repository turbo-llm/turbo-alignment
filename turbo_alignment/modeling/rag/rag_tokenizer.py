import os
from pathlib import Path

from transformers import AutoTokenizer

from turbo_alignment.settings.rag_model import RAGPreTrainedModelSettings


class RagTokenizer:
    def __init__(self, model_settings: RAGPreTrainedModelSettings, tokenizer_path: str | None, use_fast: bool = False):
        qe_config = model_settings.question_encoder_settings
        gen_config = model_settings.generator_settings

        if tokenizer_path is not None:
            question_encoder_path = Path(tokenizer_path) / Path('question_encoder_tokenizer')
            generator_path = Path(tokenizer_path) / Path('generator_tokenizer')
            self.question_encoder = AutoTokenizer.from_pretrained(question_encoder_path, use_fast=use_fast)
            self.generator = AutoTokenizer.from_pretrained(generator_path, use_fast=use_fast)
        else:
            self.question_encoder = AutoTokenizer.from_pretrained(qe_config.model_path, use_fast=use_fast)
            self.generator = AutoTokenizer.from_pretrained(gen_config.model_path, use_fast=use_fast)

        self.current_tokenizer = self.generator

        self.bos_token_id = self.generator.bos_token_id
        self.eos_token_id = self.generator.eos_token_id
        self.pad_token_id = self.generator.pad_token_id
        self.unk_token_id = self.generator.unk_token_id
        self.sep_token_id = self.generator.sep_token_id

        self.padding_side = self.generator.padding_side

    def save_pretrained(self, save_directory) -> None:
        if os.path.isfile(save_directory):
            raise ValueError(f'Provided path ({save_directory}) should be a directory, not a file')
        os.makedirs(save_directory, exist_ok=True)
        question_encoder_path = os.path.join(save_directory, 'question_encoder_tokenizer')
        generator_path = os.path.join(save_directory, 'generator_tokenizer')
        self.question_encoder.save_pretrained(question_encoder_path)
        self.generator.save_pretrained(generator_path)

    def __call__(self, *args, **kwargs):
        return self.current_tokenizer(*args, **kwargs)

    def __len__(self):
        return len(self.generator)

    def batch_decode(self, *args, **kwargs) -> list[str]:
        return self.generator.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs) -> str:
        return self.generator.decode(*args, **kwargs)

    def encode(self, *args, **kwargs) -> list[int]:
        return self.generator.encode(*args, **kwargs)

    def _switch_to_input_mode(self) -> None:
        self.current_tokenizer = self.question_encoder

    def _switch_to_target_mode(self) -> None:
        self.current_tokenizer = self.generator

    def pad(self, *args, **kwargs):
        return self.generator.pad(*args, **kwargs)

    def add_special_tokens(self, special_tokens_dict) -> int:
        added_tokens = self.generator.add_special_tokens(special_tokens_dict)
        # TODO: fix this
        if list(special_tokens_dict.keys())[0] == 'sep_token':
            self.sep_token_id = self.generator.sep_token_id
        if list(special_tokens_dict.keys())[0] == 'pad_token':
            self.pad_token_id = self.generator.pad_token_id
        return added_tokens

    def get_added_vocab(self) -> dict[str, int]:
        return self.generator.get_added_vocab()
