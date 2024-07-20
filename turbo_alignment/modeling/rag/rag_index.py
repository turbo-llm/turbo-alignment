import numpy as np
from datasets import Dataset, load_from_disk

from turbo_alignment.common.logging import get_project_logger

logger = get_project_logger()


class Index:
    """
    A base class for the Indices encapsulated by the [`RagRetriever`].
    """

    def get_doc_dicts(self, doc_ids: np.ndarray) -> list[dict]:
        """
        Returns a list of dictionaries, containing titles and text of the retrieved documents.

        Args:
            doc_ids (`np.ndarray` of shape `(batch_size, n_docs)`):
                A tensor of document indices.
        """
        raise NotImplementedError

    def get_top_docs(self, question_hidden_states: np.ndarray, n_docs=5) -> tuple[np.ndarray, np.ndarray]:
        """
        For each query in the batch, retrieves `n_docs` documents.

        Args:
            question_hidden_states (`np.ndarray` of shape `(batch_size, vector_size)`):
                An array of query vectors.
            n_docs (`int`):
                The number of docs retrieved per query.

        Returns:
            `np.ndarray` of shape `(batch_size, n_docs)`: A tensor of indices of retrieved documents. `np.ndarray` of
            shape `(batch_size, vector_size)`: A tensor of vector representations of retrieved documents.
        """
        raise NotImplementedError

    def is_initialized(self):
        """
        Returns `True` if index is already initialized.
        """
        raise NotImplementedError

    def init_index(self):
        """
        A function responsible for loading the index into memory. Should be called only once per training run of a RAG
        model. E.g. if the model is trained on multiple GPUs in a distributed setup, only one of the workers will load
        the index.
        """
        raise NotImplementedError


class HFIndexBase(Index):
    def __init__(self, vector_size, dataset, index_initialized=False) -> None:
        self.vector_size = vector_size
        self.dataset = dataset
        self._index_initialized = index_initialized
        self._check_dataset_format(with_index=index_initialized)
        dataset.set_format('numpy', columns=['embeddings'], output_all_columns=True, dtype='float32')

    def _check_dataset_format(self, with_index: bool) -> None:
        if not isinstance(self.dataset, Dataset):
            raise ValueError(f'Dataset should be a datasets.Dataset object, but got {type(self.dataset)}')
        if len({'title', 'text', 'embeddings'} - set(self.dataset.column_names)) > 0:
            raise ValueError(
                'Dataset should be a dataset with the following columns: '
                'title (str), text (str) and embeddings (arrays of dimension vector_size), '
                f'but got columns {self.dataset.column_names}'
            )
        if with_index and 'embeddings' not in self.dataset.list_indexes():
            raise ValueError(
                'Missing faiss index in the dataset. Make sure you called `dataset.add_faiss_index` to compute it '
                'or `dataset.load_faiss_index` to load one from the disk.'
            )

    def init_index(self):
        raise NotImplementedError()

    def is_initialized(self):
        return self._index_initialized

    def get_doc_dicts(self, doc_ids: np.ndarray) -> list[dict]:
        return [self.dataset[doc_ids[i].tolist()] for i in range(doc_ids.shape[0])]

    def get_top_docs(self, question_hidden_states: np.ndarray, n_docs=5) -> tuple[np.ndarray, np.ndarray]:
        _, ids = self.dataset.search_batch('embeddings', question_hidden_states, n_docs)
        docs = [self.dataset[[i for i in indices if i >= 0]] for indices in ids]
        vectors = [doc['embeddings'] for doc in docs]
        for i, vec in enumerate(vectors):
            if len(vec) < n_docs:
                vectors[i] = np.vstack([vec, np.zeros((n_docs - len(vec), self.vector_size))])
        return np.array(ids), np.array(vectors)  # shapes (batch_size, n_docs) and (batch_size, n_docs, d)


class CustomHFIndex(HFIndexBase):
    def __init__(self, vector_size: int, dataset, index_path=None) -> None:
        super().__init__(vector_size, dataset, index_initialized=index_path is None)
        self.index_path = index_path

    @classmethod
    def load_from_disk(cls, vector_size, dataset_path, index_path):
        logger.info(f'Loading passages from {dataset_path}')
        if dataset_path is None or index_path is None:
            raise ValueError(
                'Please provide `dataset_path` and `index_path` after calling `dataset.save_to_disk(dataset_path)` '
                "and `dataset.get_index('embeddings').save(index_path)`."
            )
        dataset = load_from_disk(str(dataset_path))
        return cls(vector_size=vector_size, dataset=dataset, index_path=str(index_path))

    def init_index(self) -> None:
        if not self.is_initialized():
            logger.info(f'Loading index from {self.index_path}')
            self.dataset.load_faiss_index('embeddings', file=self.index_path)
            self._index_initialized = True
