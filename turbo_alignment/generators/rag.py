import torch
from transformers import BatchEncoding

from turbo_alignment.dataset.chat.models import ChatDatasetRecord
from turbo_alignment.generators.chat import ChatGeneratorBase
from turbo_alignment.settings.generators.outputs.chat import (
    AnswerMessage,
    RagInferenceOutput,
)


class RagGenerator(ChatGeneratorBase[ChatDatasetRecord, RagInferenceOutput]):
    def generate_from_batch_records(
        self,
        dataset_name: str,
        records_batch: dict[str, torch.Tensor] | BatchEncoding,
        original_records: list[ChatDatasetRecord] | None = None,
    ) -> list[RagInferenceOutput]:
        raise ValueError('You can not use batch generation with RAG generator')

    def generate_from_single_record(
        self,
        dataset_name: str,
        record: dict[str, torch.Tensor],
        original_record: ChatDatasetRecord | None = None,
    ) -> RagInferenceOutput:
        input_ids = torch.unsqueeze(record['input_ids'], 0).to(self.device)

        answer_indices, document_indices, doc_scores = self._model.generate(
            inputs=input_ids,
            generation_config=self._transformers_generator_parameters,
            tokenizer=self._tokenizer,
            pad_token_id=self._tokenizer.pad_token_id,
        )

        answers = self._decode(token_indices=answer_indices.cpu())
        documents = self._decode(token_indices=document_indices.cpu())
        doc_scores = list(doc_scores[0])

        return RagInferenceOutput(
            id=original_record.id,
            dataset_name=dataset_name,
            messages=original_record.messages,
            label=original_record.label,
            answers=[AnswerMessage(id=str(i), content=a) for i, a in enumerate(answers)],
            documents=documents,
            doc_scores=doc_scores,
        )
