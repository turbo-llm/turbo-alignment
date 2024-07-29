import torch
import torch.nn.functional as F


def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def get_question_embeddings(encoder_output, attention_mask):
    # not sure if it works best with dpr, as it uses pooler_output
    last_hidden_state = encoder_output.hidden_states[-1]
    embeddings = average_pool(last_hidden_state, attention_mask)
    return F.normalize(embeddings, p=2, dim=1)


def pad_joined(t: torch.Tensor, max_len: int, pad_token: int = 0) -> torch.Tensor:
    return torch.cat([torch.full((1, max_len - t.shape[1]), pad_token).to(t), t], dim=1)


def pad_and_stack(tensor_list: list[torch.Tensor], max_len: int, pad_token: int) -> torch.Tensor:
    padded_tensor_list = [pad_joined(t, max_len, pad_token).squeeze(0) for t in tensor_list]
    return torch.stack(padded_tensor_list)


def join_query_and_docs(
    input_ids: torch.Tensor,
    doc_input_ids: torch.Tensor,
    pad_token_id: int,
    attention_mask: torch.Tensor | None = None,
    doc_attention_mask: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
):
    # input_ids=[1, seq_length]
    # doc_input_ids=[1, doc_length]
    inputs_list, attentions_list, labels_list = [], [], []
    max_len = 0
    for i, doc in enumerate(doc_input_ids):
        mask = torch.where(doc != pad_token_id)
        doc_inputs = doc[mask].unsqueeze(0)
        joined_inputs = torch.cat([doc_inputs, input_ids[i : i + 1]], dim=1)
        inputs_list.append(joined_inputs)
        if max_len < joined_inputs.shape[1]:
            max_len = joined_inputs.shape[1]
        if attention_mask is not None and doc_attention_mask is not None:
            doc_attn_mask = doc_attention_mask[i][mask].unsqueeze(0)
            joined_attn_mask = torch.cat([doc_attn_mask, attention_mask[i : i + 1]], dim=1)
            attentions_list.append(joined_attn_mask)
        if labels is not None:
            doc_labels = torch.full(doc_inputs.shape, -100).to(labels)
            joined_labs = torch.cat([doc_labels, labels[i : i + 1]], dim=1)
            labels_list.append(joined_labs)
    joined_input_ids = pad_and_stack(inputs_list, max_len, pad_token_id)
    joined_labels, joined_attention_mask = None, None
    if attention_mask is not None:
        joined_attention_mask = pad_and_stack(attentions_list, max_len, 0)
    if labels is not None:
        joined_labels = pad_and_stack(labels_list, max_len, -100)
    return joined_input_ids, joined_attention_mask, joined_labels
