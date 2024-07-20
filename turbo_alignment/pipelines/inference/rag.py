from typing import Generator

import torch
from accelerate import Accelerator
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from turbo_alignment.common.tf.loaders.model.model import load_model
from turbo_alignment.generators.base import BaseGenerator
from turbo_alignment.generators.rag import RagGenerator
from turbo_alignment.modeling.rag.rag_model import RagSequenceForGeneration
from turbo_alignment.modeling.rag.rag_tokenizer import RagTokenizer
from turbo_alignment.pipelines.inference.base import BaseInferenceStrategy
from turbo_alignment.settings.pipelines.inference import RAGInferenceExperimentSettings
from turbo_alignment.settings.rag_model import RAGPreTrainedModelSettings


class RAGInferenceStrategy(BaseInferenceStrategy):
    @staticmethod
    def _load_model(
        model_settings: RAGPreTrainedModelSettings,
        tokenizer: PreTrainedTokenizerBase,
    ) -> torch.nn.Module | PreTrainedModel:
        generator = load_model(model_settings.generator_settings, tokenizer.generator)
        question_encoder = load_model(model_settings.question_encoder_settings, tokenizer.question_encoder)

        model = RagSequenceForGeneration(model_settings, generator, question_encoder, tokenizer)
        return model

    def _get_single_inference_settings(
        self, experiment_settings: RAGInferenceExperimentSettings, accelerator: Accelerator
    ) -> Generator[tuple[PreTrainedTokenizerBase, BaseGenerator, str, dict], None, None]:
        save_file_id = 0

        for model_inference_settings in experiment_settings.inference_settings:
            tokenizer = RagTokenizer(
                model_settings=model_inference_settings.model_settings,
                tokenizer_path=model_inference_settings.tokenizer_settings.tokenizer_path,
                use_fast=model_inference_settings.tokenizer_settings.use_fast,
            )

            model = self._load_model(model_inference_settings.model_settings, tokenizer).to(accelerator.device)
            model.eval()

            for generation_settings in model_inference_settings.generation_settings:
                generator = RagGenerator(
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
