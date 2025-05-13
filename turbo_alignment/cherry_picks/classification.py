import math
from typing import Iterable

from accelerate import Accelerator
from transformers import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel

from turbo_alignment.cherry_picks.base import CherryPickCallbackBase
from turbo_alignment.dataset.chat.conversation import Conversation
from turbo_alignment.dataset.classification.classification import (
    InferenceClassificationDataset,
)
from turbo_alignment.generators.classification import ClassificationGenerator
from turbo_alignment.metrics.metric import Metric
from turbo_alignment.settings.cherry_pick import ClassificationCherryPickSettings
from turbo_alignment.settings.metric import ElementWiseScores, MetricResults


class ClassificationCherryPickCallback(CherryPickCallbackBase[InferenceClassificationDataset]):
    def __init__(
        self,
        cherry_pick_settings: ClassificationCherryPickSettings,
        datasets: Iterable[InferenceClassificationDataset],
        metrics: list[Metric],
    ) -> None:
        super().__init__(cherry_pick_settings=cherry_pick_settings, datasets=datasets, metrics=metrics)

    def _get_dataset_metrics(
        self,
        dataset: InferenceClassificationDataset,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        **kwargs,
    ) -> list[MetricResults]:
        accelerator: Accelerator = kwargs.get('accelerator', None)

        generator = ClassificationGenerator(
            model=model,
            tokenizer=tokenizer,
        )

        if accelerator is not None:
            dataset = self._get_sharded_dataset(
                dataset=dataset,
                accelerator=accelerator,
            )

        generations = generator.generate_from_dataset(dataset)
        predictions = [record.predicted_label for record in generations]
        labels = [record['labels'] for record in dataset]

        contexts = [
            Conversation(
                system_prompt=dataset.source.system_prompt,
                messages=record.messages,
                ignore_system_prompt=dataset.settings.chat_settings.ignore_system_prompt,
            ).get_prompt_repr(0, len(record.messages))
            for record in generations
        ]

        metric_outputs = [
            MetricResults(
                element_wise_scores=[ElementWiseScores(label=dataset.source.name + '@@' + 'contexts', values=contexts)]
            ),
            MetricResults(
                element_wise_scores=[
                    ElementWiseScores(label=dataset.source.name + '@@' + 'predictions', values=predictions)
                ]
            ),
            MetricResults(
                element_wise_scores=[ElementWiseScores(label=dataset.source.name + '@@' + 'labels', values=labels)]
            ),
        ]

        return metric_outputs

    @staticmethod
    def _get_sharded_dataset(
        dataset: InferenceClassificationDataset, accelerator: Accelerator
    ) -> InferenceClassificationDataset:
        rank_device = accelerator.process_index
        slice_size = math.ceil(len(dataset) / accelerator.num_processes)

        return dataset.get_slice(rank_device * slice_size, rank_device * slice_size + slice_size)
