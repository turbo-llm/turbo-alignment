import gc
import json
import math
import os
from typing import Dict, List

import loguru
import torch
import vllm
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from turbo_alignment.common.tf.loaders import load_model, load_tokenizer
from turbo_alignment.dataset.base import BaseDataset
from turbo_alignment.dataset.chat import ChatDatasetRecord
from turbo_alignment.dataset.loader import DatasetLoader
from turbo_alignment.generators.vllm_chat import VLLMChatGenerator
from turbo_alignment.metrics import Metric, MetricSettingsRegistry
from turbo_alignment.metrics.registry import KLType
from turbo_alignment.metrics.utils import get_logits
from turbo_alignment.pipelines.base import BaseStrategy
from turbo_alignment.settings.datasets import DatasetStrategy
from turbo_alignment.settings.pipelines.inference.metrics import MetricsSettings


class MetricsInferenceStrategy(BaseStrategy):
    def run(self, experiment_settings: MetricsSettings) -> None:
        accelerator = Accelerator()
        set_seed(seed=0, device_specific=False)
        experiment_settings.save_path.mkdir(parents=True, exist_ok=True)

        metrics = [
            Metric.by_name(metric.type)(MetricSettingsRegistry.by_name(metric.type)(**metric.parameters))
            for metric in experiment_settings.metric_settings
        ]

        model_inference_settings = experiment_settings.inference_settings
        tokenizer = load_tokenizer(
            model_inference_settings.tokenizer_settings,
            model_inference_settings.model_settings,
        )

        model = load_model(model_inference_settings.model_settings, tokenizer)
        model = accelerator.prepare_model(model, device_placement=True, evaluation_mode=True)

        sft = load_model(model_inference_settings.model_settings, tokenizer)  # fix
        sft = accelerator.prepare_model(sft, device_placement=True, evaluation_mode=True)

        vllm_model = vllm.LLM(
            model=model_inference_settings.model_settings.model_path.absolute().as_posix(),
            dtype='bfloat16',
            gpu_memory_utilization=0.3,
            tensor_parallel_size=model_inference_settings.tensor_parallel_size,
            # max_model_len=model_inference_settings.max_model_len,
        )

        dataset = DatasetLoader[BaseDataset](BaseDataset).load_datasets(
            experiment_settings.dataset_settings,
            tokenizer=tokenizer,
            strategy=DatasetStrategy.INFERENCE,
        )[0]

        generation_settings = experiment_settings.inference_settings.generation_settings[0]

        batch_size = model_inference_settings.batch
        generator = VLLMChatGenerator(
            model=vllm_model,
            tokenizer=tokenizer,
            transformers_settings=generation_settings.transformers_settings,
            custom_generation_settings=generation_settings.custom_settings,
            batch=batch_size,
            return_logits=True,
        )

        input_records = [dataset[idx] for idx in range(len(dataset))]
        records_batches = [
            input_records[i * batch_size : (i + 1) * batch_size] for i in range(math.ceil(len(dataset) / batch_size))
        ]

        original_records_batches: list[list[ChatDatasetRecord]] = [
            [dataset.get_original_record_by_id(r['id']) for r in batch] for batch in records_batches
        ]

        num_gens = generation_settings.transformers_settings.num_return_sequences

        metrics_kwargs = {}

        with open(os.path.join(experiment_settings.save_path, 'metrics.jsonl'), 'w') as f:
            for i, (records_batch, original_records_batch) in enumerate(
                zip(records_batches, original_records_batches)
            ):
                loguru.logger.info('batch {}/{}', i + 1, len(records_batches))

                generation_outputs = generator._generate_from_batch(
                    records_batch,
                    original_records_batch,
                    dataset.source.name,
                )

                string_answers = [[answer.content for answer in g.answers] for g in generation_outputs]
                string_labels = [[g.messages[-1].content] * len(g.answers) for g in generation_outputs]

                flattened_answers = [answer for g in generation_outputs for answer in g.answers]
                answer_tokens_ids = [answer.answer_token_ids.cpu() for answer in flattened_answers]
                input_tokens_ids = [answer.input_token_ids.cpu() for answer in flattened_answers]

                logits = get_logits(input_tokens_ids, answer_tokens_ids, model)
                sft_logits = get_logits(input_tokens_ids, answer_tokens_ids, sft)
                metrics_kwargs[KLType.SFT_MODEL] = sft_logits

                batch_metrics = [{} for _ in range(batch_size)]

                for i in range(len(batch_metrics)):
                    batch_metrics[i]['context'] = []
                    batch_metrics[i]['label'] = []
                    for idx in range(len(original_records_batch[i].messages[:-1])):
                        batch_metrics[i]['context'].append(
                            {
                                'content': original_records_batch[i].messages[idx].content,
                                'role': original_records_batch[i].messages[idx].role,
                            }
                        )

                    batch_metrics[i]['label'] = [
                        {
                            'content': original_records_batch[i].messages[-1].content,
                            'role': original_records_batch[i].messages[-1].role,
                        }
                    ]

                    batch_metrics[i]['answers'] = [
                        {'id': idx, 'content': string_answers[i][idx]} for idx in range(len(string_answers[i]))
                    ]

                for metric in metrics:
                    metric_results = metric.compute(
                        dataset=dataset,
                        references=string_labels,
                        predictions=string_answers,
                        accelerator=accelerator,
                        tokenizer=tokenizer,
                        dataset_name=dataset.source.name,
                        answer_tokens_ids=answer_tokens_ids,
                        input_token_ids=input_tokens_ids,
                        logits=logits,
                        metrics_kwargs=metrics_kwargs,
                    )[0].element_wise_scores
                    for scores in metric_results:
                        metric_name = scores.label.split('@@', 1)[-1]
                        metric_values = scores.values
                        if metric_name == 'reward':
                            metric_values = [m[0] for m in metric_values]
                        if metric_name in ['reward', 'kl_with_sft', 'length', 'perplexity']:
                            metric_values = [
                                metric_values[i * num_gens : (i + 1) * num_gens]
                                for i in range(len(metric_values) // num_gens)
                            ]

                        for i in range(len(metric_values)):
                            batch_metrics[i][metric_name] = metric_values[i]

                for item in batch_metrics:
                    if len(item) == 0:
                        continue
                    json.dump(item, f)
                    f.write('\n')

                del batch_metrics
                del generation_outputs
                del flattened_answers
                del string_answers
                del string_labels
                del logits
                del sft_logits
                gc.collect()
