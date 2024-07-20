from typing import Generator

import torch
from accelerate import Accelerator
from transformers import PreTrainedTokenizerBase

from turbo_alignment.common.tf.loaders import load_model, load_tokenizer
from turbo_alignment.generators.multimodal import MultimodalGenerator
from turbo_alignment.modeling.multimodal.lm.projection import (
    ProjectionMultiModalModeling,
)
from turbo_alignment.pipelines.inference.base import BaseInferenceStrategy
from turbo_alignment.pipelines.mixin.multimodal import MultimodalMixin
from turbo_alignment.settings.pipelines.inference.multimodal import (
    MultimodalInferenceExperimentSettings,
    MultimodalSingleModelInferenceSettings,
)


class MultimodalInferenceStrategy(MultimodalMixin, BaseInferenceStrategy[MultimodalInferenceExperimentSettings]):
    @staticmethod
    def _load_model(
        inference_settings: MultimodalSingleModelInferenceSettings,
        tokenizer: PreTrainedTokenizerBase,
    ):
        language_model = load_model(inference_settings.model_settings, tokenizer)

        modality_encoders = MultimodalInferenceStrategy._load_modality_encoders(
            inference_settings.modality_encoder_settings_mapping,
            device=language_model.device,
            dtype=language_model.dtype,
        )

        model = ProjectionMultiModalModeling(
            language_model=language_model,
            encoders=modality_encoders,
            n_modality_embs=inference_settings.model_settings.n_modality_embeddings,
            modality_projector_mapping=inference_settings.modality_projector_mapping,
            modality_projector_initialization_mapping=None,
            peft=True,
        )
        model.modality_adapters.load_state_dict(torch.load(inference_settings.model_settings.projections_path))

        return model

    def _get_single_inference_settings(
        self, experiment_settings: MultimodalInferenceExperimentSettings, accelerator: Accelerator
    ) -> Generator[tuple[PreTrainedTokenizerBase, MultimodalGenerator, str, dict], None, None]:
        save_file_id = 0

        for model_inference_settings in experiment_settings.inference_settings:
            tokenizer = load_tokenizer(
                model_inference_settings.tokenizer_settings,
                model_inference_settings.model_settings,
            )
            model = self._load_model(model_inference_settings, tokenizer)

            for generation_settings in model_inference_settings.generation_settings:
                generator = MultimodalGenerator(
                    transformers_settings=generation_settings.transformers_settings,
                    custom_generation_settings=generation_settings.custom_settings,
                    tokenizer=tokenizer,
                    model=model,
                    accelerator=accelerator,
                )

                parameters_to_save = {
                    'model_settings': model_inference_settings.model_settings.dict(),
                    'generation_settings': generation_settings.dict(),
                }

                yield tokenizer, generator, f'single_inference_{save_file_id}.jsonl', parameters_to_save
