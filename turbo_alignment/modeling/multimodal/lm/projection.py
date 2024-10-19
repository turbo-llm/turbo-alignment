from collections import defaultdict
from pathlib import Path

import torch
from scipy.ndimage.measurements import find_objects, label
from transformers.utils import ModelOutput

from turbo_alignment.common.logging import get_project_logger
from turbo_alignment.modeling.multimodal.lm.base import BaseMultiModalModeling
from turbo_alignment.modeling.multimodal.projectors import LlavaMultiModalProjector
from turbo_alignment.modeling.multimodal.projectors.registry import (
    MultiModalProjectorRegistry,
)
from turbo_alignment.settings.modality import Modality

logger = get_project_logger()


class ProjectionMultiModalModeling(BaseMultiModalModeling):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        modality_adapters = {}

        for modality in self.encoders:
            modality_adapters[modality] = (
                MultiModalProjectorRegistry.by_name(self.modality_projector_mapping[modality])(
                    encoder_hidden_size=self.encoders[modality].emb_dim,
                    text_hidden_size=self.language_model_dim,
                    n_modality_embs=self.n_modality_embs,
                )
                .to(self.language_model.device)
                .to(torch.bfloat16)
            )

            assert (
                not isinstance(modality_adapters[modality], LlavaMultiModalProjector)
                or self.n_modality_embs == self.encoders[modality].n_modality_embs
            ), f'For the LLaVA MLP projector, n_modality_embs should be equal to the \
total number of patches returned by the encoder. \
Please, set n_modality_embs to {self.encoders[modality].n_modality_embs} in config.'

            if self.modality_projector_initialization_mapping:
                state_dict_path: Path | None = self.modality_projector_initialization_mapping.get(modality, None)
                if state_dict_path is not None:
                    logger.info(f'Loading {modality} connector weights')

                    state_dictionary = torch.load(state_dict_path)
                    modality_adapters[modality].load_state_dict(state_dictionary)
                    logger.info(f'Sucsessfully loaded from {state_dict_path}')

        self.modality_adapters = torch.nn.ModuleDict(modality_adapters)

        self.config = self.language_model.config

    def convert_inputs_to_embeds(
        self,
        input_ids: torch.LongTensor,
        modality_inputs: list[list[tuple[Modality, torch.Tensor]]],
        modality_tokens_mask: torch.Tensor,
    ):
        multimodal_lm_input_embeds: list[torch.Tensor] = []

        lm_input_embeds = self.language_model.base_model.model.model.embed_tokens.modules_to_save.default(input_ids)

        for sample_lm_input_embeds, sample_modality_tokens_mask, sample_modality_inputs in zip(
            lm_input_embeds, modality_tokens_mask, modality_inputs
        ):
            span_mask, _ = label(
                sample_modality_tokens_mask.cpu().detach().numpy()
            )  # returns mask with ids of spans from 1 to N
            modality_spans = find_objects(span_mask)  # returns list of tuples with start index and end index

            assert len(modality_spans) == len(sample_modality_inputs)

            grouped_modality_encoder_inputs: dict[Modality, list[tuple[int, torch.Tensor]]] = defaultdict(list)

            # Prepare modality batches
            for index, modality_object in enumerate(sample_modality_inputs):
                modality, inputs = modality_object
                grouped_modality_encoder_inputs[modality].append((index, inputs))

            sorted_modality_embeddings: torch.Tensor = torch.full(
                (len(sample_modality_inputs), self.n_modality_embs, self.language_model_dim), torch.nan
            ).to(self.language_model.device)

            # Encode modalities and insert into input embeds
            for modality, modality_encoder_inputs_with_indices in grouped_modality_encoder_inputs.items():
                modality_encoder_input_indexes, modality_encoder_inputs = zip(*modality_encoder_inputs_with_indices)

                if self.language_model.dtype == torch.float32:
                    encoded_modality_object_batch = self.encoders[modality].encode(
                        torch.stack(modality_encoder_inputs, dim=0).to(self.language_model.device)
                    )
                else:
                    encoded_modality_object_batch = self.encoders[modality].encode(
                        torch.stack(modality_encoder_inputs, dim=0).to(self.language_model.device).bfloat16()
                    )

                modality_encoder_embeddings = self.modality_adapters[modality](encoded_modality_object_batch)

                sorted_modality_embeddings[modality_encoder_input_indexes, :] = modality_encoder_embeddings.to(
                    sorted_modality_embeddings.dtype
                )

            substituted_sample_lm_input_embeds = sample_lm_input_embeds.clone()
            for i, modality_span in enumerate(modality_spans):
                substituted_sample_lm_input_embeds[
                    modality_span[0].start : modality_span[0].stop
                ] = sorted_modality_embeddings[i, :]

            multimodal_lm_input_embeds.append(substituted_sample_lm_input_embeds)

        return torch.stack(multimodal_lm_input_embeds)

    def forward(
        self,
        input_ids: torch.LongTensor,
        modality_inputs: list[list[tuple[Modality, torch.Tensor]]],
        attention_mask: torch.LongTensor,
        modality_tokens_mask: torch.Tensor,
        labels: torch.LongTensor | None = None,
    ) -> ModelOutput:
        multimodal_lm_input_embeds = self.convert_inputs_to_embeds(input_ids, modality_inputs, modality_tokens_mask)
        return self.language_model(
            inputs_embeds=multimodal_lm_input_embeds, labels=labels, attention_mask=attention_mask
        )
