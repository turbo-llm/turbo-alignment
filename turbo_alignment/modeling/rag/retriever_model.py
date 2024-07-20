import time

import numpy as np
import torch
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from turbo_alignment.common.logging import get_project_logger
from turbo_alignment.modeling.rag.rag_index import CustomHFIndex
from turbo_alignment.modeling.rag.rag_tokenizer import RagTokenizer
from turbo_alignment.settings.rag_model import RAGPreTrainedModelSettings

logger = get_project_logger()


class RagRetriever:
    """
    Stores indexed documents, retrieves documents relevant for a query.
    Note: __call__ returns documents encoded with generator tokenizer.
    """

    def __init__(self, config: RAGPreTrainedModelSettings, rag_tokenizer: RagTokenizer, init_retrieval=True) -> None:
        self._init_retrieval = init_retrieval
        super().__init__()

        self.index = self._build_index(config)

        self.generator_tokenizer = rag_tokenizer.generator
        self.question_encoder_tokenizer = rag_tokenizer.question_encoder
        self.rag_tokenizer = rag_tokenizer

        self.n_docs = config.retrieval_settings.n_docs
        self.batch_size = config.retrieval_settings.retrieval_batch_size

        self.config: RAGPreTrainedModelSettings = config
        if self._init_retrieval:
            self.init_retrieval()

        self.ctx_encoder_tokenizer = None
        self.return_tokenized_docs = False

    @staticmethod
    def _build_index(config) -> CustomHFIndex:
        # CanonicalHFIndex is removed, return if needed
        return CustomHFIndex.load_from_disk(
            vector_size=config.retrieval_settings.retrieval_vector_size,
            dataset_path=config.index_settings.passages_path,
            index_path=config.index_settings.index_path,
        )

    def save_pretrained(self, save_directory) -> None:
        # self.config.save_pretrained(save_directory) FIXME
        self.rag_tokenizer.save_pretrained(save_directory)

    def init_retrieval(self) -> None:
        logger.info('initializing retrieval')
        self.index.init_index()

    def postprocess_docs(self, docs: list[dict], n_docs: int, return_tensors: str = 'pt'):
        def cat_input_and_doc(doc_title, doc_text):
            if doc_title.startswith('"'):
                doc_title = doc_title[1:]
            if doc_title.endswith('"'):
                doc_title = doc_title[:-1]
            out = (
                doc_title
                + self.config.retrieval_settings.title_sep
                + doc_text
                + self.config.retrieval_settings.doc_sep
            ).replace('  ', ' ')
            return out.strip()

        rag_input_strings = [
            cat_input_and_doc(docs[i]['title'][j], docs[i]['text'][j]) for i in range(len(docs)) for j in range(n_docs)
        ]

        contextualized_inputs = self.generator_tokenizer.batch_encode_plus(
            rag_input_strings,
            max_length=self.config.retrieval_settings.max_doc_length,
            return_tensors=return_tensors,
            padding=True,
            truncation=True,
        )

        return contextualized_inputs['input_ids'], contextualized_inputs['attention_mask']

    def _chunk_tensor(self, t, chunk_size):
        return [t[i : i + chunk_size] for i in range(0, len(t), chunk_size)]

    def _main_retrieve(self, question_hidden_states: np.ndarray, n_docs: int) -> tuple[np.ndarray, np.ndarray]:
        question_hidden_states_batched = self._chunk_tensor(question_hidden_states, self.batch_size)
        ids_batched: list[np.ndarray] = []
        vectors_batched: list[np.ndarray] = []
        for question_hidden_states_batch in question_hidden_states_batched:
            start_time = time.time()
            ids, vectors = self.index.get_top_docs(question_hidden_states_batch, n_docs)
            logger.debug(
                f'index search time: {time.time() - start_time} sec, batch size {question_hidden_states_batch.shape}'
            )
            ids_batched.extend(ids)
            vectors_batched.extend(vectors)
        return (
            np.array(ids_batched),
            np.array(vectors_batched),
        )  # shapes (batch_size, n_docs) and (batch_size, n_docs, d)

    def retrieve(self, question_hidden_states: np.ndarray, n_docs: int) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        doc_ids, retrieved_doc_embeds = self._main_retrieve(question_hidden_states, n_docs)
        return retrieved_doc_embeds, doc_ids, self.index.get_doc_dicts(doc_ids)

    def set_ctx_encoder_tokenizer(self, ctx_encoder_tokenizer: PreTrainedTokenizer) -> None:
        # used in end2end retriever training
        self.ctx_encoder_tokenizer = ctx_encoder_tokenizer
        self.return_tokenized_docs = True

    def __call__(
        self,
        question_input_ids: torch.Tensor,
        question_hidden_states: np.ndarray,
        n_docs: int,
        return_tensors=None,
    ) -> BatchEncoding:
        retrieved_doc_embeds, doc_ids, docs = self.retrieve(question_hidden_states, n_docs)

        context_input_ids, context_attention_mask = self.postprocess_docs(docs, n_docs, return_tensors=return_tensors)
        return BatchEncoding(
            {
                'context_input_ids': context_input_ids,
                'context_attention_mask': context_attention_mask,
                'retrieved_doc_embeds': retrieved_doc_embeds,
                'doc_ids': doc_ids,
            },
            tensor_type=return_tensors,
        )
