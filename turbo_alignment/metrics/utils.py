import torch
from transformers import PreTrainedModel


def calculate_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, pad_token_id: int, reduction: str):
    shift_logits = logits[:, :-1, :].contiguous().to(torch.float32)
    shift_labels = labels[:, 1:].contiguous().long()
    mask = (shift_labels != pad_token_id).to(torch.int)
    valid_length = mask.sum(dim=-1)

    ce_func = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction=reduction)
    ce = ce_func(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    ce = ce.view(shift_labels.size(0), -1)
    ce = torch.sum(ce * mask, -1) / valid_length
    return ce


def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logp = torch.nn.functional.log_softmax(logits, dim=-1)
    logps = torch.gather(logp, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return logps


def get_logits(
    input_tokens_ids: list[torch.Tensor], answer_tokens_ids: list[torch.Tensor], model: PreTrainedModel
) -> list[torch.Tensor]:
    logits = []

    for input_token_ids, answer_token_ids in zip(input_tokens_ids, answer_tokens_ids):
        with torch.no_grad():
            input_ = torch.cat((input_token_ids, answer_token_ids), dim=-1)  # calculate logits on input + answer
            logit = model(input_.to(model.device)).logits
            logit = logit[:, input_token_ids.size(-1) :, :].cpu()
            logits.append(logit)  # return only answer logits
    return logits
