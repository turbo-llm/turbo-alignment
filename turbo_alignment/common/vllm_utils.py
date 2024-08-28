from typing import List, Tuple

import torch
import vllm
from transformers import AutoTokenizer


def pad_list(
    tokenizer: AutoTokenizer,
    tensor_list: List[torch.Tensor],
    pad_side: str,
    pad_value: int,
    max_length: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    old_padding_side = tokenizer.padding_side
    old_pad_value = tokenizer.pad_token_id
    tokenizer.padding_side = pad_side
    tokenizer.pad_token_id = pad_value

    padout = tokenizer.pad(
        {"input_ids": tensor_list},
        return_tensors="pt",
        return_attention_mask=True,
        max_length=max_length,
        padding="max_length",
    )

    tokenizer.padding_side = old_padding_side
    tokenizer.pad_token_id = old_pad_value

    return padout["input_ids"], padout["attention_mask"]


def vllm_generations_postprocess(
    tokenizer: AutoTokenizer,
    generations: List[vllm.outputs.RequestOutput],
    max_length: int,
    pad_to_max_len: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Queries+responses will go the reward model. Should be padded from the left. Output from the last token.
    Queries+responses will go the reference model. Padding doesn't matter. Output from the response tokens.
    """

    # Merge queries and responses to the same sequences
    # Create a mask showing where are the response tokens
    query_responses = []
    response_tokens_mask = []
    # logprobs = []
    for g in generations:
        for i in range(len(g.outputs)):
            prompt_tokens = torch.tensor(g.prompt_token_ids)
            response_tokens = torch.tensor(g.outputs[i].token_ids)[
                : max_length - len(prompt_tokens)
            ]
            qr = torch.cat(
                [
                    prompt_tokens,
                    response_tokens,
                ],
            )
            assert len(qr.shape) == 1, qr.shape
            query_responses.append(qr)

            assert len(response_tokens) != 0, (
                len(response_tokens),
                response_tokens,
                g,
            )
            assert len(prompt_tokens) != 0, (
                len(prompt_tokens),
                prompt_tokens,
                g,
            )

            r_attn = torch.cat(
                [
                    torch.zeros(len(prompt_tokens)),
                    torch.ones(len(response_tokens)),
                ],
            )

            assert len(r_attn.shape) == 1, r_attn.shape
            response_tokens_mask.append(r_attn)

    # Query-responses are padded to the same length
    query_responses, attention_mask = pad_list(
        tokenizer,
        query_responses,
        "left",
        tokenizer.pad_token_id,
        max_length=max_length,
    )
    # as well as the response masks
    response_tokens_mask, _ = pad_list(
        tokenizer,
        response_tokens_mask,
        "left",
        0,
        max_length=max_length,
    )

    position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
    position_ids.masked_fill_(attention_mask.to(torch.bool) == 0, 0)

    return (
        query_responses,
        attention_mask.to(torch.int32),
        response_tokens_mask.to(torch.int32),
        position_ids.to(torch.int32),
    )
