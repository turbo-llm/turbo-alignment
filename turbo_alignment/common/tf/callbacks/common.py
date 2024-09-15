from collections import defaultdict
from typing import Any, Callable

import pandas as pd
from accelerate.utils.operations import gather_object
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer_callback import CallbackHandler

from turbo_alignment.settings.metric import MetricResults


class EvaluateFirstStepCallback(TrainerCallback):
    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step == 0:
            control.should_evaluate = True

        return control


class MetricsCallbackHandler(CallbackHandler):
    def __init__(self, *args, **kwargs) -> None:
        self.ref_model = kwargs.pop('ref_model', None)
        self.sft_model = kwargs.pop('sft_model', None)
        self.accelerator = kwargs.pop('accelerator', None)
        self.metrics_kwargs = kwargs.pop('metrics_kwargs', {})
        super().__init__(*args, **kwargs)

    def on_evaluate(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics
    ) -> TrainerControl:
        control.should_evaluate = False

        results = self.call_event(
            'on_evaluate',
            args,
            state,
            control,
            metrics=metrics,
            ref_model=self.ref_model,
            sft_model=self.sft_model,
            accelerator=self.accelerator,
            metrics_kwargs=self.metrics_kwargs,
        )
        if isinstance(results, list):
            gathered_results: list[list[MetricResults]] = gather_object(results)

            gathered_float_data: dict[str, list[Any]] = defaultdict(list)
            gathered_table_data: dict[str, list[str] | list[list[str]]] = defaultdict(list)
            average_functions: dict[str, Callable] = {}

            for single_process_results in gathered_results:
                for metric_result in single_process_results:
                    for score in metric_result.element_wise_scores:
                        if metric_result.need_average:
                            gathered_float_data[score.label].extend(score.values)
                            average_functions[score.label] = score.average_function
                        else:
                            gathered_table_data[score.label].extend(score.values)

            logs = {
                'cherry_pick_' + k: average_functions[k](list(*zip(*v)))
                if isinstance(v[0], tuple)
                else average_functions[k](v)
                for k, v in gathered_float_data.items()
            }

            self.call_event('on_log', args, state, control, logs=logs)

            table_cols = list(gathered_table_data.keys())
            table_data = list(gathered_table_data.values())

            flattened_table_data = [
                sum(item, []) if isinstance(item, list) and isinstance(item[0], list) else item for item in table_data
            ]  # flatten list[lists] to display all outputs in wandb table

            cherrypicks_table_data = pd.DataFrame(columns=table_cols, data=list(zip(*flattened_table_data)))
            dataset_prefixes = set(col.split('@@')[0] for col in cherrypicks_table_data.columns)

            for dataset_prefix in dataset_prefixes:
                dataset_columns = [col for col in cherrypicks_table_data.columns if col.startswith(dataset_prefix)]

                table = {
                    f'cherry_pick_table_{dataset_prefix}_{state.global_step}': cherrypicks_table_data[dataset_columns]
                }

                self.call_event('on_log', args, state, control, logs=table)

        return control
