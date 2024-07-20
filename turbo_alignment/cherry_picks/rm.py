from typing import Iterable

from accelerate import Accelerator
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from turbo_alignment.cherry_picks.base import CherryPickCallbackBase
from turbo_alignment.dataset.pair_preferences import PairPreferenceDataset
from turbo_alignment.generators.rm import RMPairGenerator
from turbo_alignment.metrics.metric import Metric
from turbo_alignment.settings.cherry_pick import RMCherryPickSettings
from turbo_alignment.settings.metric import ElementWiseScores, MetricResults


class RmCherryPickCallback(CherryPickCallbackBase[PairPreferenceDataset]):
    def __init__(
        self,
        cherry_pick_settings: RMCherryPickSettings,
        datasets: Iterable[PairPreferenceDataset],
        metrics: list[Metric],
    ) -> None:
        super().__init__(cherry_pick_settings=cherry_pick_settings, datasets=datasets, metrics=metrics)

    def _get_dataset_metrics(
        self,
        dataset: PairPreferenceDataset,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        **kwargs,
    ) -> list[MetricResults]:
        accelerator: Accelerator = kwargs.get('accelerator', None)

        generator = RMPairGenerator(
            model=model,
            tokenizer=tokenizer,
            accelerator=accelerator,
        )

        generations = generator.generate_from_dataset(dataset)
        generations_w = [gen.reward_w for gen in generations]
        generations_l = [gen.reward_l for gen in generations]

        pair_scores = [1 if w > l else 0 for w, l in zip(generations_w, generations_l)]

        answers_w = [record.answer_w.content for record in generations]
        answers_l = [record.answer_l.content for record in generations]

        metric_outputs = [
            MetricResults(
                element_wise_scores=[ElementWiseScores(label=dataset.source.name + '@@' + 'chosen', values=answers_w)]
            ),
            MetricResults(
                element_wise_scores=[
                    ElementWiseScores(label=dataset.source.name + '@@' + 'rejected', values=answers_l)
                ]
            ),
            MetricResults(
                element_wise_scores=[
                    ElementWiseScores(label=dataset.source.name + '@@' + 'chosen_reward', values=generations_w)
                ]
            ),
            MetricResults(
                element_wise_scores=[
                    ElementWiseScores(label=dataset.source.name + '@@' + 'rejected_reward', values=generations_l)
                ]
            ),
            MetricResults(
                element_wise_scores=[ElementWiseScores(label=dataset.source.name + '@@' + 'score', values=pair_scores)]
            ),
        ]

        return metric_outputs
