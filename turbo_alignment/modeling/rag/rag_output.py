from dataclasses import dataclass

import torch
from transformers.modeling_outputs import ModelOutput


@dataclass
class RetrieveAugLMOutput(ModelOutput):
    # RetrieveAugLMOutput should not have more than one required field.
    logits: torch.FloatTensor
    labels: torch.Tensor | None = None
    doc_scores: torch.Tensor | None = None
    past_key_values: list[torch.FloatTensor] | None = None
    retrieved_doc_embeds: torch.FloatTensor | None = None
    retrieved_doc_ids: torch.LongTensor | None = None
    joined_input_ids: torch.Tensor | None = None
    joined_attention_mask: torch.Tensor | None = None
    question_encoder_last_hidden_state: torch.FloatTensor | None = None
    question_enc_hidden_states: tuple[torch.FloatTensor] | None = None
    question_enc_attentions: tuple[torch.FloatTensor] | None = None
    generator_enc_last_hidden_state: torch.FloatTensor | None = None
    generator_enc_hidden_states: tuple[torch.FloatTensor] | None = None
    generator_enc_attentions: tuple[torch.FloatTensor] | None = None
    generator_dec_hidden_states: tuple[torch.FloatTensor] | None = None
    generator_dec_attentions: tuple[torch.FloatTensor] | None = None
    generator_cross_attentions: tuple[torch.FloatTensor] | None = None


@dataclass
class RetrieveAugLMMarginOutput(ModelOutput):
    # RetrieveAugLMMarginOutput should not have more than one required field.
    loss: torch.Tensor
    logits: torch.FloatTensor | None = None
    doc_scores: torch.Tensor | None = None
    past_key_values: list[torch.FloatTensor] | None = None
    retrieved_doc_embeds: torch.FloatTensor | None = None
    retrieved_doc_ids: torch.LongTensor | None = None
    context_input_ids: torch.LongTensor | None = None
    context_attention_mask: torch.LongTensor | None = None
    question_encoder_last_hidden_state: torch.FloatTensor | None = None
    question_enc_hidden_states: tuple[torch.FloatTensor] | None = None
    question_enc_attentions: tuple[torch.FloatTensor] | None = None
    generator_enc_last_hidden_state: torch.FloatTensor | None = None
    generator_enc_hidden_states: tuple[torch.FloatTensor] | None = None
    generator_enc_attentions: tuple[torch.FloatTensor] | None = None
    generator_dec_hidden_states: tuple[torch.FloatTensor] | None = None
    generator_dec_attentions: tuple[torch.FloatTensor] | None = None
    generator_cross_attentions: tuple[torch.FloatTensor] | None = None
