import typing as tp
from functools import lru_cache

import torch

from turbo_alignment.common.logging import get_project_logger

if tp.TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


logger = get_project_logger()


# adapted from
# https://github.com/InternLM/xtuner/blob/90192ffe42612b0f88409432e7b4860294432bcc/xtuner/parallel/sequence/data_collate.py#L7
def pad_for_sequence_parallel(tensor, seq_parallel_world_size, padding_value, dim=-1, padding_side='right'):
    length = tensor.shape[dim]
    if length % seq_parallel_world_size == 0:
        return tensor

    pad_num = seq_parallel_world_size - (length % seq_parallel_world_size)
    pad_shape = (
        (*tensor.shape[:dim], pad_num, *tensor.shape[dim + 1 :]) if dim != -1 else (*tensor.shape[:dim], pad_num)
    )
    pad = torch.full(pad_shape, padding_value, dtype=tensor.dtype, device=tensor.device)

    lst_to_pad = [tensor, pad] if padding_side == 'right' else [pad, tensor]
    tensor = torch.cat(lst_to_pad, dim=dim)
    return tensor


def pad_and_slice(
    tensor: torch.Tensor,
    seq_parallel_world_size: int,
    rank: int,
    padding_value: tp.Any,
    dim=-1,
    padding_side='right',
) -> torch.Tensor:
    padded = pad_for_sequence_parallel(tensor, seq_parallel_world_size, padding_value, dim, padding_side)
    chunk_size = padded.size(dim) // seq_parallel_world_size
    return padded.narrow(dim, rank * chunk_size, chunk_size)


DEFAULT_PAD_VALUES = {
    'attention_mask': False,
    'input_ids': 0,
    'labels': -100,
}


def tensor_dim_slice(tensor, dim, s):
    return tensor[(slice(None),) * (dim if dim >= 0 else dim + tensor.dim()) + (s,)]


@lru_cache
def log_once_unused_keys(keys: tuple[str, ...]):
    logger.info(f'Cannot split values, got keys: {" ".join(keys)}')


class DataCollatorForSequenceParallism:
    def __init__(
        self,
        base_collate_fn,
        seq_p_rank: int,
        seq_p_world_size: int,
        fields_not_to_split: list[str] | None = None,
        pad_values_for_fields: dict[str, tp.Any] | None = None,
        add_cache_positions: bool = True,
        padding_side: str = 'left',
    ):
        self.base_collate_fn = base_collate_fn
        self.seq_p_rank = seq_p_rank
        self.seq_p_world_size = seq_p_world_size
        self.fields_not_to_split = fields_not_to_split or ['attention_mask']
        self.pad_values_for_fields = pad_values_for_fields or DEFAULT_PAD_VALUES
        self.add_cache_positions = add_cache_positions
        self.padding_side = padding_side

    @staticmethod
    def pad_values_from_tokenizer(tokenizer: 'PreTrainedTokenizerBase') -> dict[str, tp.Any]:
        pad_values = DEFAULT_PAD_VALUES.copy()
        pad_values['input_ids'] = tokenizer.pad_token_id
        return pad_values

    @classmethod
    def create_with_tokenizer(
        cls,
        base_collate_fn,
        seq_p_rank: int,
        seq_p_world_size: int,
        tokenizer: 'PreTrainedTokenizerBase',
        fields_not_to_split: list[str] | None = None,
        add_cache_positions: bool = True,
    ):
        return cls(
            base_collate_fn,
            seq_p_rank,
            seq_p_world_size,
            fields_not_to_split=fields_not_to_split,
            pad_values_for_fields=cls.pad_values_from_tokenizer(tokenizer),
            add_cache_positions=add_cache_positions,
            padding_side=tokenizer.padding_side,
        )

    def _get_cache_position(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.arange(0, input_ids.shape[1], device=input_ids.device)

    def __call__(self, *args, **kwargs):
        collated = self.base_collate_fn(*args, **kwargs)
        if isinstance(collated, tp.Mapping):
            if 'input_ids' not in collated:
                log_once_unused_keys(tuple(collated.keys()))
                return collated

            input_ids = collated['input_ids']
            cache_position = None
            if self.add_cache_positions:
                cache_position = self._get_cache_position(input_ids)
                collated['cache_position'] = cache_position

            return {key: self.prepare_value(key, value) for key, value in collated.items()}

        return self._split_value(collated)

    def should_be_splitted(self, key):
        return key not in self.fields_not_to_split

    def prepare_value(self, key: str, value: torch.Tensor, pad_value=None):
        padded = pad_for_sequence_parallel(
            value,
            self.seq_p_world_size,
            pad_value if pad_value is not None else self.pad_values_for_fields.get(key, 0),
            dim=-1,
            padding_side=self.padding_side,
        )
        if self.should_be_splitted(key):
            if not isinstance(padded, torch.Tensor):
                raise ValueError(f'{key=} {value=} {padded=}')

            return self._split_value(padded)

        return padded

    def _split_value(self, value: torch.Tensor, dim: int = -1) -> torch.Tensor:
        start, end = self._get_slice_for_length(value.size(dim))
        return tensor_dim_slice(value, dim, slice(start, end))

    def _get_slice_for_length(self, length: int) -> tuple[int, int]:
        assert (
            length % self.seq_p_world_size == 0
        ), f'Expect length {length} divide by world size {self.seq_p_world_size}'
        chunk_size = length // self.seq_p_world_size
        return chunk_size * self.seq_p_rank, chunk_size * (self.seq_p_rank + 1)
