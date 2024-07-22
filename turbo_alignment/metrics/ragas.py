# pylint: skip-file
# pylint: disable-all
# mypy: ignore-errors

from datasets import Dataset
from ragas import RunConfig, evaluate
from ragas.metrics import (
    answer_relevancy,
    answer_similarity,
    context_entity_recall,
    context_precision,
    context_recall,
    faithfulness,
)

from turbo_alignment.dataset.chat import InferenceChatDataset
from turbo_alignment.metrics.metric import Metric
from turbo_alignment.metrics.registry import RagasMetricsSettings
from turbo_alignment.settings.generators.outputs.chat import RagInferenceOutput
from turbo_alignment.settings.metric import MetricResults, MetricType


@Metric.register(MetricType.RAGAS_METRICS)
class RagasMetrics(Metric):
    def __init__(self, settings: RagasMetricsSettings) -> None:
        self._settings: RagasMetricsSettings = settings

        if self._settings.openai_api_key is not None:
            # use openai endpoints if api key is provided
            from langchain_openai import OpenAI, OpenAIEmbeddings

            self._llm = OpenAI(openai_api_key=self._settings.openai_api_key, model="gpt-3.5-turbo-instruct")
            self._embeddings = OpenAIEmbeddings(
                openai_api_key=self._settings.openai_api_key, model="text-embedding-3-large"
            )

        elif self._settings.mistralai_api_key is not None:
            from langchain_mistralai import MistralAIEmbeddings
            from langchain_mistralai.chat_models import ChatMistralAI

            self._llm = ChatMistralAI(name='mistral-large', api_key=self._settings.mistralai_api_key)

            self._embeddings = MistralAIEmbeddings(api_key=self._settings.mistralai_api_key)

    def compute(
        self, dataset: InferenceChatDataset, generations: list[RagInferenceOutput], **kwargs
    ) -> list[MetricResults]:
        questions = [d['messages'][0].content for d in dataset]
        ground_truths = [d['messages'][1].content for d in dataset]
        retieved_docs = [g.documents for g in generations]
        answers = [g.answers[0].content for g in generations]

        ragas_dataset = Dataset.from_dict(
            {'question': questions, 'ground_truth': ground_truths, 'contexts': retieved_docs, 'answer': answers}
        )

        extra_kwargs = {}
        if self._llm:
            extra_kwargs['llm'] = self._llm
        if self._embeddings:
            extra_kwargs['embeddings'] = self._embeddings

        results = evaluate(
            ragas_dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                answer_similarity,
                context_precision,
                context_recall,
                context_entity_recall,
            ],
            **extra_kwargs,
            raise_exceptions=False,
            is_async=True,
            run_config=RunConfig(max_workers=1, max_wait=180, thread_timeout=600),
        )

        return results
