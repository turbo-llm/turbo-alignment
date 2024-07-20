from typing import Generator

from accelerate import Accelerator
from transformers import PreTrainedTokenizerBase

from turbo_alignment.common.tf.loaders import load_model, load_tokenizer
from turbo_alignment.generators.classification import ClassificationGenerator
from turbo_alignment.pipelines.inference.base import BaseInferenceStrategy
from turbo_alignment.settings.pipelines.inference.base import (
    InferenceExperimentSettings,
)


class ClassificationInferenceStrategy(BaseInferenceStrategy[InferenceExperimentSettings]):
    def _get_single_inference_settings(
        self, experiment_settings: InferenceExperimentSettings, accelerator: Accelerator
    ) -> Generator[tuple[PreTrainedTokenizerBase, ClassificationGenerator, str, dict], None, None]:
        save_file_id = 0

        for model_inference_settings in experiment_settings.inference_settings:
            tokenizer = load_tokenizer(
                model_inference_settings.tokenizer_settings,
                model_inference_settings.model_settings,
            )
            model = load_model(model_inference_settings.model_settings, tokenizer)
            model.eval()

            generator = ClassificationGenerator(
                tokenizer=tokenizer,
                model=model,
                accelerator=accelerator,
                batch=model_inference_settings.batch,
            )
            parameters_to_save = {'model_settings': model_inference_settings.model_settings.dict()}

            yield tokenizer, generator, f'single_inference_{save_file_id}.jsonl', parameters_to_save
