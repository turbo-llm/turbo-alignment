from transformers import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from turbo_alignment.common.logging import get_project_logger
from turbo_alignment.settings.tf.special_tokens_setter import SpecialTokensSettings

logger = get_project_logger()


class SpecialTokensSetter:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, special_tokens_settings: SpecialTokensSettings) -> None:
        self._tokenizer = tokenizer
        self._special_tokens_settings = special_tokens_settings
        self._special_tokens_already_set: bool = False

    def setBOS(self, bos_token: str) -> None:
        if self._tokenizer.bos_token_id is None:
            logger.info('Model does not have bos_token_id')
            self._tokenizer.add_special_tokens(special_tokens_dict={'bos_token': bos_token})
            assert self._tokenizer.bos_token_id is not None
            logger.info(f'Created bos_token_id = {self._tokenizer.bos_token_id}')
        else:
            logger.info(f'Model has bos_token_id = {self._tokenizer.bos_token_id}')

    def setEOS(self, eos_token: str) -> None:
        if self._tokenizer.eos_token_id is None:
            logger.info('Model does not have eos_token_id')
            self._tokenizer.add_special_tokens(special_tokens_dict={'eos_token': eos_token})
            assert self._tokenizer.eos_token_id is not None
            logger.info(f'Created eos_token_id = {self._tokenizer.eos_token_id}')
        else:
            logger.info(f'Model has eos_token_id = {self._tokenizer.eos_token_id}')

    def setPAD(self, pad_token: str | None) -> None:
        if self._tokenizer.pad_token_id is None and pad_token is None:
            logger.info('Skip adding pad_token_id')
            return None

        if self._tokenizer.pad_token_id is None:
            logger.info('Model does not have pad_token_id')
            self._tokenizer.add_special_tokens(special_tokens_dict={'pad_token': pad_token})
            assert self._tokenizer.pad_token_id is not None
            logger.info(f'Created pad_token_id = {self._tokenizer.pad_token_id}')
        else:
            logger.info(f'Model has pad_token_id = {self._tokenizer.pad_token_id}')

        return None

    def setUNK(self, unk_token: str | None) -> None:
        if self._tokenizer.unk_token_id is None and unk_token is None:
            logger.info('Skip adding sep_token_id')
            return None

        if self._tokenizer.unk_token_id is None:
            logger.info('Model does not have unk_token_id')
            self._tokenizer.add_special_tokens(special_tokens_dict={'unk_token': unk_token})
            assert self._tokenizer.unk_token_id is not None
            logger.info(f'Created unk_token_id = {self._tokenizer.unk_token_id}')
        else:
            logger.info(f'Model has unk_token_id = {self._tokenizer.unk_token_id}')

        return None

    def setSEP(self, sep_token: str | None) -> None:
        if self._tokenizer.sep_token_id is None and sep_token is None:
            logger.info('Skip adding sep_token_id')
            return None

        if self._tokenizer.sep_token_id is None:
            logger.info('Model does not have sep_token_id')
            self._tokenizer.add_special_tokens(special_tokens_dict={'sep_token': sep_token})
            assert self._tokenizer.sep_token_id is not None
            logger.info(f'Created sep_token_id = {self._tokenizer.sep_token_id}')
        else:
            logger.info(f'Model has sep_token_id = {self._tokenizer.sep_token_id}')

        return None

    def set_all(self) -> None:
        self.setBOS(bos_token=self._special_tokens_settings.bos_token)
        self.setEOS(eos_token=self._special_tokens_settings.eos_token)
        self.setPAD(pad_token=self._special_tokens_settings.pad_token)
        self.setUNK(unk_token=self._special_tokens_settings.unk_token)
        self.setSEP(sep_token=self._special_tokens_settings.sep_token)

    def set_custom_tokens(self, tokens: list[str]) -> None:
        if self._special_tokens_already_set:
            raise ValueError('Additional special tokens already set')

        self._special_tokens_already_set = True

        logger.info(f'Added custom special tokens: {tokens}')
        added_tokens = self._tokenizer.add_special_tokens({'additional_special_tokens': tokens})
        assert added_tokens == len(tokens)

    def setup_model_config(self, model: PreTrainedModel) -> None:
        model.config.bos_token_id = self._tokenizer.bos_token_id
        model.config.eos_token_id = self._tokenizer.eos_token_id

        if self._tokenizer.pad_token_id is not None:
            model.config.pad_token_id = self._tokenizer.pad_token_id
        if self._tokenizer.sep_token_id is not None:
            model.config.sep_token_id = self._tokenizer.sep_token_id
