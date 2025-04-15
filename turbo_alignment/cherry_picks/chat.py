import math
from typing import Iterable

from accelerate import Accelerator
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from turbo_alignment.cherry_picks.base import CherryPickCallbackBase
from turbo_alignment.dataset.chat import InferenceChatDataset
from turbo_alignment.generators.chat import ChatGenerator
from turbo_alignment.metrics.metric import Metric
from turbo_alignment.metrics.registry import KLType
from turbo_alignment.metrics.utils import get_logits
from turbo_alignment.modeling import parallel_states
from turbo_alignment.settings.cherry_pick import ChatCherryPickSettings
from turbo_alignment.settings.metric import ElementWiseScores, MetricResults

import torch.distributed as dist
from turbo_alignment.dist_utils.order import print_in_order, run_in_order


class ChatCherryPickCallback(CherryPickCallbackBase[InferenceChatDataset]):
    def __init__(
        self,
        cherry_pick_settings: ChatCherryPickSettings,
        datasets: Iterable[InferenceChatDataset],
        metrics: list[Metric],
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        super().__init__(
            cherry_pick_settings=cherry_pick_settings,
            datasets=datasets,
            metrics=metrics,
            tokenizer=tokenizer,
        )
        self._custom_generation_settings = cherry_pick_settings.custom_generation_settings
        self._generator_transformers_settings = cherry_pick_settings.generator_transformers_settings

    def _get_dataset_metrics(
        self,
        dataset: InferenceChatDataset,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        **kwargs,
    ) -> list[MetricResults]:
        accelerator: Accelerator = kwargs.get('accelerator', None)
        metrics_kwargs: dict = kwargs.get('metrics_kwargs', {})
        ref_model: dict = kwargs.get('ref_model', None)
        sft_model: dict = kwargs.get('sft_model', None)

        generator = ChatGenerator(
            model=model,
            tokenizer=tokenizer,
            transformers_settings=self._generator_transformers_settings,
            custom_generation_settings=self._custom_generation_settings,
            return_logits=True,
        )

        batch_size = self._generator_transformers_settings.num_return_sequences

        @run_in_order()
        def print_dataset(prefix, dataset):
            print(f'{prefix} {dist.get_rank()=} {dataset[0]=}')

        print_dataset('Before sharding:', dataset)

        if accelerator is not None:
            dataset = self._get_sharded_dataset(
                dataset=dataset,
                accelerator=accelerator,
            )

        print_dataset('After sharding:', dataset)

        generations = generator.generate_from_dataset(dataset)

        prompts = [record['prompt'] for record in dataset]
        string_answers = [[answer.content for answer in g.answers] for g in generations]
        string_labels = [[g.messages[-1].content] * len(g.answers) for g in generations]

        flattened_answers = [answer for g in generations for answer in g.answers]

        answer_tokens_ids = [answer.answer_token_ids.cpu() for answer in flattened_answers]
        input_tokens_ids = [answer.input_token_ids.cpu() for answer in flattened_answers]

        logits = [answer.logits[:, answer.input_token_ids.size(-1) :, :].cpu() for answer in flattened_answers]

        if ref_model is not None:
            metrics_kwargs[KLType.REFERENCE_MODEL] = get_logits(input_tokens_ids, answer_tokens_ids, ref_model)

        if sft_model is not None:
            metrics_kwargs[KLType.SFT_MODEL] = get_logits(input_tokens_ids, answer_tokens_ids, sft_model)

        metric_outputs = [
            MetricResults(
                element_wise_scores=[
                    ElementWiseScores(
                        label=dataset.source.name + '@@' + 'prompt',
                        values=[prompt for prompt in prompts for _ in range(batch_size)],
                    )
                ]
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
                accelerator=accelerator,
                logits=logits,
                answer_tokens_ids=answer_tokens_ids,
                tokenizer=tokenizer,
                input_token_ids=input_tokens_ids,
                dataset_name=dataset.source.name,
                metrics_kwargs=metrics_kwargs,
            )

            metric_outputs.extend(metric_results)
        return metric_outputs

    @staticmethod
    def _get_sharded_dataset(dataset: InferenceChatDataset, accelerator: Accelerator) -> InferenceChatDataset:
        rank_device = accelerator.process_index
        world_size = accelerator.num_processes
        if parallel_states.sequence_parallel_is_enabled():
            rank_device = parallel_states.get_data_parallel_rank()
            world_size = parallel_states.get_data_parallel_world_size()

        print_in_order(None)(f'{dist.get_rank()=} {rank_device=} {world_size=}')
        slice_size = math.ceil(len(dataset) / world_size)

        return dataset.get_slice(rank_device * slice_size, rank_device * slice_size + slice_size)
