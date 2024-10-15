from typing import Iterable

import torch
from PIL import Image
from transformers import PreTrainedTokenizerBase

import wandb
from turbo_alignment.cherry_picks.base import CherryPickCallbackBase
from turbo_alignment.dataset.multimodal import InferenceMultimodalDataset
from turbo_alignment.generators.multimodal import MultimodalGenerator
from turbo_alignment.metrics.metric import Metric
from turbo_alignment.settings.cherry_pick import MultimodalCherryPickSettings
from turbo_alignment.settings.metric import ElementWiseScores, MetricResults


class MultimodalCherryPickCallback(CherryPickCallbackBase[InferenceMultimodalDataset]):
    def __init__(
        self,
        cherry_pick_settings: MultimodalCherryPickSettings,
        datasets: Iterable[InferenceMultimodalDataset],
        metrics: list[Metric],
    ) -> None:
        super().__init__(cherry_pick_settings=cherry_pick_settings, datasets=datasets, metrics=metrics)
        self._custom_generation_settings = cherry_pick_settings.custom_generation_settings
        self._generator_transformers_settings = cherry_pick_settings.generator_transformers_settings

    def _get_dataset_metrics(
        self,
        dataset: InferenceMultimodalDataset,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        **kwargs,
    ) -> list[MetricResults]:
        generator = MultimodalGenerator(
            model=model,
            tokenizer=tokenizer,
            transformers_settings=self._generator_transformers_settings,
            custom_generation_settings=self._custom_generation_settings,
        )

        generations = generator.generate_from_dataset(dataset)

        metrics_kwargs: dict = kwargs.get('metrics_kwargs', {})

        prompts = [dataset[i]['prompt'] for i in range(len(dataset))]
        string_answers = [[answer.content for answer in g.answers] for g in generations]
        string_labels = [[g.messages[-1].content] * len(g.answers) for g in generations]

        modality_object_paths_batch = [dataset[i]['modality_object_paths'] for i in range(len(dataset))]

        wandb_images: list[wandb.Image | None] = []
        wandb_audios: list[wandb.Audio | None] = []
        wandb_paths: list[str | None] = []

        for path_batch in modality_object_paths_batch:
            if len(path_batch) > 0:
                path = path_batch[0]  # TODO: How to display multiple images in wandb?
                if path.split('.')[-1] == 'jpg':
                    try:
                        with Image.open(path) as img:
                            wandb_images.append(wandb.Image(img, caption=f'Path {path}'))
                    except FileNotFoundError:
                        wandb_images.append(None)
                    wandb_audios.append(None)
                elif path.split('.')[-1] in ('wav', 'flac'):
                    wandb_audios.append(wandb.Audio(path, caption=f'Path {path}'))
                    wandb_images.append(None)
                else:
                    wandb_images.append(None)
                    wandb_audios.append(None)
                wandb_paths.append(path)
            else:
                wandb_paths.append(None)
                wandb_images.append(None)
                wandb_audios.append(None)

        metric_outputs = [
            MetricResults(
                element_wise_scores=[
                    ElementWiseScores(label=dataset.source.name + '@@' + 'image', values=wandb_images)
                ]
            ),
            MetricResults(
                element_wise_scores=[
                    ElementWiseScores(label=dataset.source.name + '@@' + 'audio', values=wandb_audios)
                ]
            ),
            MetricResults(
                element_wise_scores=[
                    ElementWiseScores(label=dataset.source.name + '@@' + 'modality_object_path', values=wandb_paths)
                ]
            ),
            MetricResults(
                element_wise_scores=[ElementWiseScores(label=dataset.source.name + '@@' + 'prompt', values=prompts)]
            ),
            MetricResults(
                element_wise_scores=[
                    ElementWiseScores(label=dataset.source.name + '@@' + 'labels', values=string_labels)
                ]
            ),
            MetricResults(
                element_wise_scores=[
                    ElementWiseScores(label=dataset.source.name + '@@' + 'answer', values=string_answers)
                ]
            ),
        ]

        for metric in self._metrics:
            metric_results = metric.compute(
                model=model,
                dataset=dataset,
                references=string_labels,
                predictions=string_answers,
                tokenizer=tokenizer,
                dataset_name=dataset.source.name,
                metrics_kwargs=metrics_kwargs,
            )

            metric_outputs.extend(metric_results)

        return metric_outputs
