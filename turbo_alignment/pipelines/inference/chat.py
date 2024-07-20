from typing import Generator

import torch
from accelerate import Accelerator
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from turbo_alignment.common.tf.loaders import load_model, load_tokenizer
from turbo_alignment.generators.base import BaseGenerator
from turbo_alignment.generators.chat import ChatGenerator
from turbo_alignment.pipelines.inference.base import BaseInferenceStrategy
from turbo_alignment.settings.pipelines.inference.chat import (
    ChatInferenceExperimentSettings,
)


class ChatInferenceStrategy(BaseInferenceStrategy[ChatInferenceExperimentSettings]):
    def _get_single_inference_settings(
        self, experiment_settings: ChatInferenceExperimentSettings, accelerator: Accelerator
    ) -> Generator[tuple[PreTrainedTokenizerBase, BaseGenerator, str, dict], None, None]:
        save_file_id = 0

        for model_inference_settings in experiment_settings.inference_settings:
            tokenizer = load_tokenizer(
                model_inference_settings.tokenizer_settings,
                model_inference_settings.model_settings,
            )

            if model_inference_settings.use_vllm:
                import vllm

                from turbo_alignment.generators.vllm_chat import VLLMChatGenerator

                model = vllm.LLM(
                    model=model_inference_settings.model_settings.model_path.absolute().as_posix(),
                    dtype='bfloat16',
                    tensor_parallel_size=model_inference_settings.tensor_parallel_size,
                )
            else:
                model: PreTrainedModel = load_model(model_inference_settings.model_settings, tokenizer)
                model = (
                    accelerator.prepare_model(model, device_placement=True, evaluation_mode=True)
                    if torch.cuda.is_available()
                    else model.to('cpu')
                )

            for generation_settings in model_inference_settings.generation_settings:
                generator_kwargs = {
                    'model': model,
                    'tokenizer': tokenizer,
                    'transformers_settings': generation_settings.transformers_settings,
                    'custom_generation_settings': generation_settings.custom_settings,
                    'batch': model_inference_settings.batch,
                }
                generator = (
                    ChatGenerator(
                        **generator_kwargs,
                        accelerator=accelerator,
                    )
                    if not model_inference_settings.use_vllm
                    else VLLMChatGenerator(**generator_kwargs)
                )

                parameters_to_save = {
                    'model_settings': model_inference_settings.model_settings.dict(),
                    'generation_settings': generation_settings.dict(),
                }

                save_file_id += 1

                yield tokenizer, generator, f'single_inference_{save_file_id}.jsonl', parameters_to_save
