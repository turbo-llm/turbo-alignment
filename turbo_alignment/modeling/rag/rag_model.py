import torch
from torch import nn
from torch.nn import Module
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    GenerationConfig,
    PreTrainedModel,
)

from turbo_alignment.common.logging import get_project_logger
from turbo_alignment.modeling.rag.rag_output import (
    RetrieveAugLMMarginOutput,
    RetrieveAugLMOutput,
)
from turbo_alignment.modeling.rag.rag_tokenizer import RagTokenizer
from turbo_alignment.modeling.rag.retriever_model import RagRetriever
from turbo_alignment.modeling.rag.utils import (
    get_question_embeddings,
    join_query_and_docs,
)
from turbo_alignment.settings.rag_model import RAGPreTrainedModelSettings

logger = get_project_logger()


class RagModel(Module):
    def __init__(
        self,
        generator: AutoModelForCausalLM,
        question_encoder: AutoModel,
        config: RAGPreTrainedModelSettings,
        tokenizer: RagTokenizer,
    ):
        super().__init__()
        self.rag_config = config

        self.generator = generator
        self.question_encoder = question_encoder
        self.retriever = RagRetriever(config=config, rag_tokenizer=tokenizer)

        self.ctx_encoder = None
        self.context_encoder_training = False
        self.tokenizer = tokenizer

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> RetrieveAugLMOutput:
        input_ids = input_ids.to(self.generator.device)
        labels = labels.to(self.generator.device)

        # decode query with generator tokenizer
        decoded_inputs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        # TODO HACK
        decoded_inputs = [i.split('<RS><bot>')[0] for i in decoded_inputs]

        decoded_inputs = [f'{self.rag_config.retrieval_settings.prefix} {inp}' for inp in decoded_inputs]
        # encode query with retriever tokenizer
        retriever_tokenizer_output = self.tokenizer.question_encoder(
            decoded_inputs,
            max_length=self.rag_config.retrieval_settings.query_encoder_max_length,
            padding=True,
            truncation=True,
            return_tensors='pt',
        )
        retriever_input_ids = retriever_tokenizer_output['input_ids'].to(self.question_encoder.device)
        retriever_attn_mask = retriever_tokenizer_output['attention_mask'].to(self.question_encoder.device)

        # get query embeddings
        question_enc_outputs = self.question_encoder(
            retriever_input_ids,
            attention_mask=retriever_attn_mask,
            return_dict=True,
            output_hidden_states=True,
        )
        question_encoder_last_hidden_state = get_question_embeddings(question_enc_outputs, retriever_attn_mask)

        # retrieve ONE document, it will already be tokenized with generator tokenizer
        retriever_outputs = self.retriever(
            retriever_input_ids,
            question_encoder_last_hidden_state.cpu().detach().to(torch.float32).numpy(),
            n_docs=self.rag_config.retrieval_settings.n_docs,
            return_tensors='pt',
        )
        doc_input_ids, doc_attention_mask, retrieved_doc_embeds, retrieved_doc_ids = (
            retriever_outputs['context_input_ids'],
            retriever_outputs['context_attention_mask'],
            retriever_outputs['retrieved_doc_embeds'],
            retriever_outputs['doc_ids'],
        )
        doc_input_ids = doc_input_ids.to(input_ids)
        doc_attention_mask = doc_attention_mask.to(input_ids)

        # join document and query input ids
        joined_input_ids, joined_attention_mask, joined_labels = join_query_and_docs(
            input_ids, doc_input_ids, self.tokenizer.pad_token_id, attention_mask, doc_attention_mask, labels
        )

        # set to correct device
        retrieved_doc_embeds = retrieved_doc_embeds.to(question_encoder_last_hidden_state)
        joined_input_ids = joined_input_ids.to(input_ids)
        joined_attention_mask = joined_attention_mask.to(input_ids)

        # compute doc_scores
        doc_scores = torch.bmm(
            question_encoder_last_hidden_state.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)
        ).squeeze(1)

        gen_outputs = self.generator(
            input_ids=joined_input_ids,
            attention_mask=joined_attention_mask,
            return_dict=True,
        )

        question_enc_hidden_states = question_enc_outputs.hidden_states
        question_enc_attentions = question_enc_outputs.attentions

        return RetrieveAugLMOutput(
            logits=gen_outputs.logits,
            labels=joined_labels,
            doc_scores=doc_scores,
            past_key_values=gen_outputs.past_key_values,
            joined_input_ids=joined_input_ids,
            joined_attention_mask=joined_attention_mask,
            retrieved_doc_embeds=retrieved_doc_embeds,
            retrieved_doc_ids=retrieved_doc_ids,
            question_encoder_last_hidden_state=question_encoder_last_hidden_state,
            question_enc_hidden_states=question_enc_hidden_states,
            question_enc_attentions=question_enc_attentions,
            generator_dec_hidden_states=gen_outputs.hidden_states,
            generator_dec_attentions=gen_outputs.attentions,
        )


class RagSequenceForGeneration(Module):
    def __init__(
        self,
        model_settings: RAGPreTrainedModelSettings,
        generator: AutoModelForCausalLM,
        question_encoder: AutoModel,
        tokenizer: RagTokenizer,
    ):
        '''
        Implementation of RAG-Sequence from the paper:
        Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
        https://arxiv.org/pdf/2005.11401
        '''
        super().__init__()
        self.rag_config = model_settings
        self.generation_config = GenerationConfig
        self.tokenizer = tokenizer
        self.rag = RagModel(
            config=self.rag_config, generator=generator, question_encoder=question_encoder, tokenizer=tokenizer
        )

        # this config is needed for Trainer to work, rag_config is not PretrainedConfig
        self.config = self.rag.generator.config

        self.device = self.rag.generator.device

    def set_retriever(self, retriever: RagRetriever) -> None:
        self.rag.retriever = retriever

    def set_context_encoder_for_training(self, ctx_encoder: PreTrainedModel) -> None:
        self.rag.context_encoder_training = True
        self.rag.ctx_encoder = ctx_encoder

    def resize_token_embeddings(self, emb_size: int) -> None:
        self.rag.generator.resize_token_embeddings(emb_size)
        self.rag.question_encoder.resize_token_embeddings(len(self.tokenizer.question_encoder))

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        **_kwargs,
    ) -> RetrieveAugLMMarginOutput:
        outputs = self.rag(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        logits = outputs.logits
        labels = outputs.labels
        joined_input_ids = outputs.joined_input_ids

        loss = self.get_nll(
            logits,
            outputs.doc_scores,
            target=joined_input_ids,
            labels=labels,
            n_docs=self.rag_config.retrieval_settings.n_docs,
        )

        out = RetrieveAugLMMarginOutput(
            loss=loss,
            logits=logits,
            doc_scores=outputs.doc_scores,
            past_key_values=outputs.past_key_values,
            context_input_ids=outputs.joined_input_ids,
            context_attention_mask=outputs.joined_attention_mask,
            retrieved_doc_embeds=outputs.retrieved_doc_embeds,
            retrieved_doc_ids=outputs.retrieved_doc_ids,
            question_encoder_last_hidden_state=outputs.question_encoder_last_hidden_state,
            question_enc_hidden_states=outputs.question_enc_hidden_states,  # nones from here
            question_enc_attentions=outputs.question_enc_attentions,
            generator_enc_last_hidden_state=outputs.generator_enc_last_hidden_state,
            generator_enc_hidden_states=outputs.generator_enc_hidden_states,
            generator_enc_attentions=outputs.generator_enc_attentions,
            generator_dec_hidden_states=outputs.generator_dec_hidden_states,
            generator_dec_attentions=outputs.generator_dec_attentions,
            generator_cross_attentions=outputs.generator_cross_attentions,
        )
        return out

    def get_nll(
        self,
        seq_logits: torch.Tensor,
        doc_scores: torch.Tensor,
        target: torch.Tensor,
        labels: torch.Tensor,
        n_docs: int,
    ) -> torch.Tensor:
        '''
        seq_logits: [batch_size*n_docs, seq_len, vocab_size]
        doc_score: [batch_size, n_docs]
        targets: [batch_size*n_docs, seq_len]
        labels: [batch_size*n_docs, seq_len]
        '''

        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/rag/modeling_rag.py#L1057

        batch_size = seq_logits.shape[0] // n_docs

        def _mask_pads(ll):
            ignore_index = -100
            pad_mask = labels.eq(ignore_index)
            if pad_mask.any():
                ll.masked_fill_(pad_mask, 0.0)
            return ll.squeeze(-1)

        # seq_logits dim = (batch*n_docs, seq_len , vocab_size)
        seq_logprobs = nn.functional.log_softmax(seq_logits, dim=-1).view(
            batch_size, n_docs, -1, seq_logits.size(-1)
        )  # batch_size , n_docs , seq_len , vocab_size

        doc_logprobs = nn.functional.log_softmax(doc_scores, dim=-1)

        # calculate loss
        seq_logprobs = seq_logprobs.flatten(0, 1)  # batch_size*n_docs , seq_len , vocab_size
        target = target.unsqueeze(-1)
        assert target.dim() == seq_logprobs.dim()

        seq_logprobs = seq_logprobs.gather(dim=-1, index=target)

        seq_logprobs = seq_logprobs.squeeze(-1)

        seq_logprobs_masked = _mask_pads(seq_logprobs)

        # sum over tokens, exclude bos while scoring
        seq_logprobs_masked = seq_logprobs_masked.reshape((batch_size, n_docs, -1))  # batch_size, n_docs, seq_len
        seq_ll = seq_logprobs_masked.sum(2)  # batch_size, n_docs

        # marginalize over docs
        ll = seq_ll + doc_logprobs

        ll = ll.logsumexp(1)  # logsumexp over docs

        loss = -ll.mean()

        return loss

    @property
    def retriever(self) -> RagRetriever:
        return self.rag.retriever

    @property
    def generator(self):
        return self.rag.generator

    @property
    def question_encoder(self):
        return self.rag.question_encoder

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.generator.prepare_inputs_for_generation(*args, **kwargs)

    @torch.no_grad()
    def generate(
        self,
        inputs: torch.LongTensor,
        generation_config: GenerationConfig | None = None,
        **kwargs,
    ) -> tuple[torch.LongTensor, torch.LongTensor, torch.Tensor]:
        # TODO remove code duplicate with forward

        inputs = inputs.to(self.question_encoder.device)

        # decode query with generator tokenizer
        decoded_inputs = self.tokenizer.batch_decode(inputs)
        # encode query with retriever tokenizer
        retriever_tokenizer_output = self.tokenizer.question_encoder(
            decoded_inputs,
            max_length=self.rag_config.retrieval_settings.query_encoder_max_length,
            padding=True,
            truncation=True,
            return_tensors='pt',
        )
        retriever_input_ids = retriever_tokenizer_output['input_ids'].to(self.question_encoder.device)
        retriever_attn_mask = retriever_tokenizer_output['attention_mask'].to(self.question_encoder.device)

        question_encoder_output = self.question_encoder(
            retriever_input_ids, attention_mask=retriever_attn_mask, output_hidden_states=True
        )
        question_hidden_states = get_question_embeddings(question_encoder_output, retriever_attn_mask)

        retriever_output = self.retriever(
            retriever_input_ids,
            question_hidden_states.cpu().detach().to(torch.float32).numpy(),
            n_docs=self.rag_config.retrieval_settings.n_docs,
            return_tensors='pt',
        )

        # this in unnecessary, computing doc scores only for wandb table
        retrieved_doc_embeds = retriever_output['retrieved_doc_embeds'].to(question_hidden_states)
        doc_scores = torch.bmm(question_hidden_states.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)).squeeze(1)

        doc_input_ids = retriever_output['context_input_ids'].to(self.question_encoder.device)
        joined_input_ids, _, _ = join_query_and_docs(
            input_ids=inputs, doc_input_ids=doc_input_ids, pad_token_id=self.tokenizer.pad_token_id
        )

        batch_size = retriever_input_ids.shape[0]
        assert batch_size == 1, 'Can not generate using batches for now'

        output_sequences = self.generator.generate(
            input_ids=joined_input_ids,
            generation_config=generation_config,
            pad_token_id=self.tokenizer.pad_token_id,
            tokenizer=kwargs.get('tokenizer', None),
        )
        # TODO chose max-prob sequence with accounting for doc probs
        only_answer_output = output_sequences[:, joined_input_ids.shape[-1] :]
        return only_answer_output, doc_input_ids, doc_scores
