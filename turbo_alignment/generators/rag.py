from typing import Any

import torch

from turbo_alignment.dataset.chat.models import ChatDatasetRecord
from turbo_alignment.generators.chat import ChatGenerator
from turbo_alignment.settings.generators.outputs.chat import (
    AnswerMessage,
    RagInferenceOutput,
)


class RagGenerator(ChatGenerator):
    def _generate_from_single_record(
        self,
        record: dict[str, Any],
        original_record: ChatDatasetRecord,
        dataset_name: str,
    ) -> RagInferenceOutput:
        input_ids = torch.unsqueeze(record['input_ids'], 0).to(self.device)

        answer_indices, document_indices, doc_scores = self._model.generate(
            inputs=input_ids,
            generation_config=self._transformers_generator_parameters,
            tokenizer=self._tokenizer.current_tokenizer,
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
