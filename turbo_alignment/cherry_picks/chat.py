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
from turbo_alignment.settings.cherry_pick import ChatCherryPickSettings
from turbo_alignment.settings.metric import ElementWiseScores, MetricResults


class ChatCherryPickCallback(CherryPickCallbackBase[InferenceChatDataset]):
    def __init__(
        self,
        cherry_pick_settings: ChatCherryPickSettings,
        datasets: Iterable[InferenceChatDataset],
        metrics: list[Metric],
    ) -> None:
        super().__init__(cherry_pick_settings=cherry_pick_settings, datasets=datasets, metrics=metrics)
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
            # accelerator=accelerator,
            return_logits=True,
        )

        batch_size = self._generator_transformers_settings.num_return_sequences

        if accelerator is not None:
            len_records_batches = len(dataset)
            world_size = accelerator.num_processes
            rank_device = accelerator.process_index
            window_size = math.ceil(len_records_batches / world_size)

            print(f"len records_batches", len_records_batches)
            print("rank_device", rank_device)
            print("world_size", world_size)
            print(f"slice [{rank_device * window_size} : {rank_device * window_size + window_size}]")


            dataset = dataset[rank_device * window_size : rank_device * window_size + window_size]

        generations = generator.generate_from_dataset(dataset)

        print(f"len dataset {len(dataset)}")
        print(f"first 2 prompts: {[record['prompt'] for record in dataset][:2]}")
        print(f"len generations {len(generations)}")
        print(f"first 2 generations: {generations[:2]}")

        prompts = [record['prompt'] for record in dataset]

        # len_records_batches = len(prompts)
        # world_size = accelerator.num_processes
        # rank_device = accelerator.process_index
        # window_size = math.ceil(len_records_batches / world_size)

        # prompts =



        string_answers = [[answer.content for answer in g.answers] for g in generations]
        print(f"prompts: {len(prompts)}, {prompts[:2]}")
        print(f"string_answers: {len(string_answers)}, {string_answers[:2]}")
        string_labels = [[g.messages[-1].content] * len(g.answers) for g in generations]

        flattened_answers = [answer for g in generations for answer in g.answers]

        answer_tokens_ids = [answer.answer_token_ids.cpu() for answer in flattened_answers]
        input_tokens_ids = [answer.input_token_ids.cpu() for answer in flattened_answers]

        logits = [answer.logits[:, answer.input_token_ids.size(-1) :, :].cpu() for answer in flattened_answers]

        if ref_model is not None:
            metrics_kwargs[KLType.REFERENCE_MODEL] = get_logits(input_tokens_ids, answer_tokens_ids, ref_model)

        if sft_model is not None:
            metrics_kwargs[KLType.SFT_MODEL] = get_logits(input_tokens_ids, answer_tokens_ids, sft_model)

        print(f"len prompts {len(prompts)}")
        print(f"batch_size {batch_size}")
        print(f"prompt element_wise_scores {len([prompt for prompt in prompts for _ in range(batch_size)])}")
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
        print(f"prompts: {len(metric_outputs[0].element_wise_scores[0].values)}, {metric_outputs[0].element_wise_scores[0].values}")
        print(f"labels: {len(metric_outputs[1].element_wise_scores[0].values)}, {metric_outputs[1].element_wise_scores[0].values}")
        # print(f"metric_outputs: {metric_outputs}")


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
