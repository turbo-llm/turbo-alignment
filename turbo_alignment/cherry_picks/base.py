from abc import abstractmethod
from typing import Generic, Iterable, TypeVar

from accelerate import Accelerator
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from turbo_alignment.dataset.base import BaseDataset
from turbo_alignment.metrics.metric import Metric
from turbo_alignment.settings.cherry_pick import CherryPickSettings
from turbo_alignment.settings.metric import MetricResults

InferenceDatasetT = TypeVar('InferenceDatasetT', bound=BaseDataset)


class CherryPickCallbackBase(TrainerCallback, Generic[InferenceDatasetT]):
    def __init__(
        self,
        cherry_pick_settings: CherryPickSettings,
        datasets: Iterable[InferenceDatasetT],
        metrics: list[Metric],
        tokenizer: PreTrainedTokenizerBase,
    ):
        super().__init__()
        self._cherry_pick_settings = cherry_pick_settings
        self._metrics = metrics
        self._datasets = datasets
        self.tokenizer = tokenizer

    @abstractmethod
    def _get_dataset_metrics(
        self,
        dataset: InferenceDatasetT,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        **kwargs,
    ) -> list[MetricResults]:
        ...

    def on_evaluate(
        self,
        _args: TrainingArguments,
        state: TrainerState,
        _control: TrainerControl,
        **kwargs,
    ) -> list[list[MetricResults]]:
        tokenizer: PreTrainedTokenizerBase = kwargs.pop('processing_class', None)
        if tokenizer is None:
            raise ValueError('Tokenizer is None')

        model: PreTrainedModel = kwargs.pop('model', None)
        if model is None:
            raise ValueError('Model is None')

        model.eval()

        dataset_metrics = []

        for dataset in self._datasets:
            dataset_metrics.append(
                self._get_dataset_metrics(
                    dataset=dataset,
                    model=model,
                    tokenizer=tokenizer,
                    **kwargs,
                ),
            )

        model.train()

        return dataset_metrics

    @staticmethod
    @abstractmethod
    def _get_sharded_dataset(dataset: InferenceDatasetT, accelerator: Accelerator) -> InferenceDatasetT:
        ...
