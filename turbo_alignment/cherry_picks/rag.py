from accelerate import Accelerator
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from turbo_alignment.cherry_picks.chat import ChatCherryPickCallback
from turbo_alignment.dataset.chat import InferenceChatDataset
from turbo_alignment.generators.rag import RagGenerator
from turbo_alignment.settings.metric import ElementWiseScores, MetricResults


class RagCherryPickCallback(ChatCherryPickCallback):
    def _get_dataset_metrics(
        self,
        dataset: InferenceChatDataset,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        **kwargs,
    ) -> list[MetricResults]:
        accelerator: Accelerator = kwargs.get('accelerator', None)

        generator = RagGenerator(
            model=model,
            tokenizer=tokenizer,
            transformers_settings=self._generator_transformers_settings,
            custom_generation_settings=self._custom_generation_settings,
            accelerator=accelerator,
        )

        generations = generator.generate_from_dataset(dataset)

        prompts = [dataset[i]['prompt'] for i in range(len(dataset))]
        answers, docs, scores = [], [], []
        for gen in generations:
            ans_group_enum = [f'{ind+1}: {ans.content}' for ind, ans in enumerate(gen.answers)]
            doc_group_enum = [f'{ind+1}: {doc}' for ind, doc in enumerate(gen.documents)]
            scores_group_enum = [f'{ind+1}: {round(s, 2)}' for ind, s in enumerate(gen.doc_scores)]
            answers.append('\n\n'.join(ans_group_enum))
            docs.append('\n\n'.join(doc_group_enum))
            scores.append('\n\n'.join(scores_group_enum))

        metric_outputs = [
            MetricResults(
                element_wise_scores=[ElementWiseScores(label=dataset.source.name + '@@' + 'prompt', values=prompts)]
            ),
            MetricResults(
                element_wise_scores=[ElementWiseScores(label=dataset.source.name + '@@' + 'answers', values=answers)]
            ),
            MetricResults(
                element_wise_scores=[ElementWiseScores(label=dataset.source.name + '@@' + 'documents', values=docs)]
            ),
            MetricResults(
                element_wise_scores=[ElementWiseScores(label=dataset.source.name + '@@' + 'scores', values=scores)]
            ),
        ]

        for metric in self._metrics:
            metric_results = metric.compute(dataset=dataset, generations=generations)
            metric_outputs.extend(metric_results)

        return metric_outputs
