import torch
from transformers import (
    DataCollatorWithPadding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from turbo_alignment.dataset.chat import InferenceChatDataset
from turbo_alignment.metrics.metric import Metric
from turbo_alignment.metrics.registry import RetrievalUtilitySettings
from turbo_alignment.metrics.utils import calculate_cross_entropy
from turbo_alignment.settings.metric import ElementWiseScores, MetricResults, MetricType


@Metric.register(MetricType.RETRIEVAL_UTILITY)
class RetrievalUtilityMetric(Metric):
    last_tokens_string: str | None = None

    def __init__(self, settings: RetrievalUtilitySettings) -> None:
        super().__init__(settings=settings)
        self._settings: RetrievalUtilitySettings = settings
        self.doc_sep_symbol = settings.doc_sep_symbol

    def compute(self, **kwargs) -> list[MetricResults]:
        model: PreTrainedModel = kwargs.get('model', None)
        tokenizer: PreTrainedTokenizerBase = kwargs.get('tokenizer', None)

        input_ids_list: list[torch.Tensor] = kwargs.get('input_token_ids', None)
        # references - ground truth question answers
        references: list[list[str]] = kwargs.get('references', None)
        dataset: InferenceChatDataset = kwargs.get('dataset', None)
        dataset_name: str = kwargs.get('dataset_name', '')

        if references is None:
            raise ValueError('references should not be None')

        if model is None:
            raise ValueError('model should not be None')

        if tokenizer is None:
            raise ValueError('tokenizer should not be None')

        if input_ids_list is None:
            raise ValueError('input_ids should not be None')

        if dataset is None:
            raise ValueError('dataset should not be None')

        retrieval_utility_scores = []

        device = model.device

        try:
            prompt_template = dataset.settings.prompt_template
            formatted_prefix = prompt_template.prefix_template.format(role=prompt_template.role_tag_mapping['bot'])
            self.last_tokens_string = f'{prompt_template.suffix_template} {formatted_prefix}'
        except (AttributeError, KeyError) as e:
            raise ValueError(
                'prompt_template should have prefix_template, suffix_template, and bot_role as spec_token'
            ) from e

        for input_ids, ref in zip(input_ids_list, references):
            true_ref = ref[0]

            input_ids = input_ids.to(device)
            segments = self._split_input_ids(tokenizer, input_ids, device)
            data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt')
            batch = data_collator([{'input_ids': segment} for segment in segments])
            segments_padded = batch['input_ids'].to(device)
            attention_mask_segments_padded = torch.where(segments_padded != tokenizer.pad_token_id, 1, 0).to(device)

            labels = tokenizer.encode(true_ref, return_tensors='pt').to(device)  # тут только ответ бота
            labels = labels.repeat(segments_padded.shape[0], 1)

            with torch.no_grad():
                inputs = torch.cat((segments_padded, labels), dim=1)
                attention_mask = torch.cat((attention_mask_segments_padded, torch.ones_like(labels)), dim=1)
                logits = model(inputs, attention_mask=attention_mask).logits

            labels = self._padding(labels, segments_padded, tokenizer, device)

            segments_ce = (
                calculate_cross_entropy(logits, labels, tokenizer.pad_token_id, reduction='none')
                .detach()
                .cpu()
                .tolist()
            )

            only_query_loss = segments_ce[0]
            retrieval_utility = [only_query_loss - x for x in segments_ce[1:]]
            retrieval_utility_scores.append(retrieval_utility)

        # понадобится, если будем фильтровать заретривленные доки по скору
        max_length = max(len(sublist) for sublist in retrieval_utility_scores)

        transposed_lists = [
            [sublist[i] if i < len(sublist) else 0 for sublist in retrieval_utility_scores] for i in range(max_length)
        ]

        metric_results_list = []

        for i, values in enumerate(transposed_lists, start=0):
            if i == 0:
                label = f'{dataset_name}@@docsAll_ReU'
            else:
                label = f'{dataset_name}@@docs{i}_ReU'

            metric_results_list.extend(
                [
                    MetricResults(
                        element_wise_scores=[ElementWiseScores(label=label, values=values)],
                        need_average=need_average,
                    )
                    for need_average in self._settings.need_average
                ]
            )

        return metric_results_list

    def _padding(self, labels: torch.Tensor, segment, tokenizer, device):
        padding = torch.full(
            [segment.shape[0], segment.shape[1]],
            fill_value=tokenizer.pad_token_id,
            dtype=torch.long,
        ).to(device)

        labels = torch.cat((padding, labels), dim=1)

        return labels

    def _split_input_ids(self, tokenizer, input_ids, device):
        segments = []

        last_tokens = torch.tensor(tokenizer.encode(self.last_tokens_string, add_special_tokens=False)).to(device)

        doc_sep_token_id = tokenizer.convert_tokens_to_ids(self.doc_sep_symbol)
        input_ids_flat = input_ids.squeeze()

        sep_positions = list((input_ids_flat == doc_sep_token_id).nonzero(as_tuple=True)[0].cpu())
        input_ids_question = input_ids_flat[: sep_positions[0]]

        segments.append(torch.cat((input_ids_question, last_tokens)))
        segments.append(input_ids_flat)

        # вопрос + док_i + special_tokens
        input_ids_question_docs = [
            torch.cat(
                (
                    input_ids_question,
                    input_ids_flat[sep_positions[i] : sep_positions[i + 1]],
                    last_tokens,
                )
            )
            for i in range(len(sep_positions) - 1)
        ]

        input_ids_question_docs.append(torch.cat((input_ids_question, input_ids_flat[sep_positions[-1] :])))

        segments.extend(input_ids_question_docs)
        return segments
