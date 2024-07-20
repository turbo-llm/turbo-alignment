from typing import Generator

from accelerate import Accelerator
from transformers import PreTrainedTokenizerBase

from turbo_alignment.common.tf.loaders import load_model, load_tokenizer
from turbo_alignment.generators.rm import RMSamplingGenerator
from turbo_alignment.pipelines.inference.base import BaseInferenceStrategy
from turbo_alignment.settings.pipelines.inference.base import (
    InferenceExperimentSettings,
)


class RMInferenceStrategy(BaseInferenceStrategy[InferenceExperimentSettings]):
    def _get_single_inference_settings(
        self, experiment_settings: InferenceExperimentSettings, accelerator: Accelerator
    ) -> Generator[tuple[PreTrainedTokenizerBase, RMSamplingGenerator, str, dict], None, None]:
        save_file_id = 0

        for model_inference_settings in experiment_settings.inference_settings:
            tokenizer = load_tokenizer(
                model_inference_settings.tokenizer_settings,
                model_inference_settings.model_settings,
            )
            model = load_model(model_inference_settings.model_settings, tokenizer).to(accelerator.device)
            model.eval()

            generator_kwargs = {'model': model, 'tokenizer': tokenizer, 'accelerator': accelerator}
            generator = RMSamplingGenerator(
                **generator_kwargs,
                batch=model_inference_settings.batch,
                micro_batch=model_inference_settings.micro_batch,
            )
            parameters_to_save = {'model_settings': model_inference_settings.model_settings.dict()}

            yield tokenizer, generator, f'single_inference_{save_file_id}.jsonl', parameters_to_save
