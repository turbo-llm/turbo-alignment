from typing import Iterable

from accelerate import Accelerator
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from turbo_alignment.cherry_picks.base import CherryPickCallbackBase
from turbo_alignment.dataset.chat.conversation import Conversation
from turbo_alignment.dataset.classification.classification import ClassificationDataset
from turbo_alignment.generators.classification import ClassificationGenerator
from turbo_alignment.metrics.metric import Metric
from turbo_alignment.settings.cherry_pick import ClassificationCherryPickSettings
from turbo_alignment.settings.metric import ElementWiseScores, MetricResults


class ClassificationCherryPickCallback(CherryPickCallbackBase[ClassificationDataset]):
    def __init__(
        self,
        cherry_pick_settings: ClassificationCherryPickSettings,
        datasets: Iterable[ClassificationDataset],
        metrics: list[Metric],
    ) -> None:
        super().__init__(cherry_pick_settings=cherry_pick_settings, datasets=datasets, metrics=metrics)

    def _get_dataset_metrics(
        self,
        dataset: ClassificationDataset,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        **kwargs,
    ) -> list[MetricResults]:
        accelerator: Accelerator = kwargs.get('accelerator', None)

        generator = ClassificationGenerator(
            model=model,
            tokenizer=tokenizer,
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
