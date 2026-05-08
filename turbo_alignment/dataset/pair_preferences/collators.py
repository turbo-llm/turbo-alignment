from typing import Any

import torch
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerBase

from turbo_alignment.constants import DISABLE_LOSS_LABEL


class PairPreferenceDataCollator(DataCollatorForSeq2Seq):
    """
    Data collator using sequential packing: [context | chosen | rejected].

    Packs all segments into a single sequence for efficient processing. Attention masks
    enforce isolation between chosen/rejected segments. Position IDs are symmetric
    (rejected mirrors chosen) for fair comparison.

    Args:
        tokenizer: Tokenizer for padding operations
        add_labels: Whether to add labels (kept for API compatibility)
        pad_to_multiple_of: Pad sequences to multiple of this value
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        add_labels: bool = True,
        pad_to_multiple_of: int | None = None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors='pt',
            label_pad_token_id=DISABLE_LOSS_LABEL,
        )
        self.add_labels = add_labels

    def _process_example(self, ex: dict) -> tuple[torch.Tensor, tuple[int, int, int], float | None]:
        """
        Pack example into sequential format: [context | chosen | rejected].

        Returns:
            - input_ids: Concatenated tensor
            - boundaries: Tuple (context_end, chosen_end, rejected_end)
            - precomputed_margin: Optional margin value
        """
        context, chosen, rejected = ex['inputs_context'], ex['inputs_chosen'], ex['inputs_rejected']

        input_ids = torch.cat([context, chosen, rejected])

        boundaries = (
            len(context),  # context_end
            len(context) + len(chosen),  # chosen_end
            len(context) + len(chosen) + len(rejected),  # rejected_end
        )

        return input_ids, boundaries, ex.get('precomputed_margin')

    def _get_attn_mask(self, boundaries_tensor: torch.Tensor, max_seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create 4D attention mask enforcing chosen/rejected isolation.

        Uses causal mask with rectangle exclusion.

        Returns:
            Tensor[batch_size, 1, max_seq_len, max_seq_len] where 1 = attend, 0 = mask
        """
        batch_size = boundaries_tensor.shape[0]

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool, device=device))
        mask = mask[None, None].expand(batch_size, 1, -1, -1).clone()

        positions = torch.arange(max_seq_len, device=device)
        row_idx = positions.view(1, 1, -1, 1)
        col_idx = positions.view(1, 1, 1, -1)

        context_grid = boundaries_tensor[:, 0].view(-1, 1, 1, 1)
        chosen_grid = boundaries_tensor[:, 1].view(-1, 1, 1, 1)
        rejected_grid = boundaries_tensor[:, 2].view(-1, 1, 1, 1)

        # Exclude positions where rejected segment (rows) attends to chosen segment (cols)
        rejected_mask = (
            (row_idx >= chosen_grid) & (row_idx < rejected_grid) & (col_idx >= context_grid) & (col_idx < chosen_grid)
        )

        # Apply exclusion in-place for memory efficiency
        mask &= ~rejected_mask

        return mask

    def _get_position_ids(self, boundaries_tensor: torch.Tensor, max_seq_len: int) -> torch.Tensor:
        """
        Compute symmetric position IDs: rejected segment mirrors chosen positions.

        Position scheme: context [0..N-1], chosen [N..N+M-1], rejected [N..N+K-1].

        Returns:
            Tensor[batch_size, max_seq_len] position IDs
        """
        batch_size = boundaries_tensor.shape[0]
        ctx_ends, chosen_ends, rejected_ends = (
            boundaries_tensor[:, 0],
            boundaries_tensor[:, 1],
            boundaries_tensor[:, 2],
        )

        base_positions = torch.arange(max_seq_len, dtype=torch.long)
        position_ids = base_positions.unsqueeze(0).expand(batch_size, max_seq_len).clone()

        positions_grid = base_positions.unsqueeze(0).expand(batch_size, -1)  # [batch_size, max_seq_len]

        rejected_mask = (positions_grid >= chosen_ends.unsqueeze(1)) & (positions_grid < rejected_ends.unsqueeze(1))

        # For each position in rejected segment, compute: ctx_end + (pos - chosen_end)
        offset_positions = ctx_ends.unsqueeze(1) + (positions_grid - chosen_ends.unsqueeze(1))

        # Apply offset positions only to rejected segments
        position_ids = torch.where(rejected_mask, offset_positions, position_ids)

        return position_ids

    def __call__(
        self, examples: list[dict[str, Any]], return_tensors=None
    ) -> dict[str, Any]:  # type: ignore[override]
        """
        Collate batch using sequential packing: [context | chosen | rejected].

        Returns:
            - 'input_ids': Padded sequences
            - 'attention_mask': 4D masks with chosen/rejected isolation
            - 'position_ids': Symmetric positions.
            - 'chosen_indices': Last token positions for chosen segments
            - 'rejected_indices': Last token positions for rejected segments
        """

        processed = [self._process_example(ex) for ex in examples]
        concat_input_ids, boundaries, _ = zip(*processed)

        # Use parent class for padding
        batch = super().__call__([{'input_ids': input_ids} for input_ids in concat_input_ids])
        device = batch['input_ids'].device

        _, max_seq_len = batch['input_ids'].shape[:2]
        boundaries_tensor = torch.tensor(boundaries, dtype=torch.long, device=device)  # [batch_size, 3]

        batch['attention_mask'] = self._get_attn_mask(boundaries_tensor, max_seq_len, device)
        batch['position_ids'] = self._get_position_ids(boundaries_tensor, max_seq_len)

        batch['context_end_indices'] = boundaries_tensor[:, 0] - 1  # Last token of context
        batch['chosen_indices'] = boundaries_tensor[:, 1] - 1
        batch['rejected_indices'] = boundaries_tensor[:, 2] - 1

        return batch
