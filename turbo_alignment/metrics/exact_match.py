import json
from datetime import datetime, timedelta
from turbo_alignment.dataset.chat import InferenceChatDataset
from turbo_alignment.metrics.metric import Metric
from turbo_alignment.metrics.registry import ExactMatchMetricSettings
from turbo_alignment.settings.metric import ElementWiseScores, MetricResults, MetricType


@Metric.register(MetricType.EXACT_MATCH)
class ExactMatchMetric(Metric):
    def __init__(self, settings: ExactMatchMetricSettings) -> None:
        super().__init__(settings=settings)
        self._settings: ExactMatchMetricSettings = settings

    def compute(self, **kwargs) -> list[MetricResults]:
        references: InferenceChatDataset = kwargs.get('references', None)
        predictions: list[list[str]] = kwargs.get('predictions', None)

        if references is None:
            raise ValueError('references should not be None')

        if predictions is None:
            raise ValueError('predictions should not be None')

        (
            ref_functions,
            ref_arguments,
        ) = self._parse_function_with_arguments(references)
        (
            pred_functions,
            pred_arguments,
        ) = self._parse_function_with_arguments(predictions)

        element_wise_scores = []
        for selected_key in self._settings.selected_argument_keys:
            if 'date' in selected_key:
                for day_lift in self._settings.possible_day_lifts:
                    temp_hits = []
                    for pred_argument, ref_argument in zip(
                            pred_arguments, ref_arguments):
                        temp_hits.append(
                            self._full_match_argument(
                                parsed_predicted_arguments=pred_argument,
                                parsed_reference_arguments=ref_argument,
                                selected_key=selected_key,
                                day_lift=day_lift
                            )
                        )
                    label = selected_key +'_'+str(day_lift)
                    element_wise_scores.append(ElementWiseScores(label=label, values=temp_hits))
            else:
                temp_hits = []
                for pred_argument, ref_argument in zip(
                        pred_arguments, ref_arguments):
                    temp_hits.append(
                        self._full_match_argument(
                            parsed_predicted_arguments=pred_argument,
                            parsed_reference_arguments=ref_argument,
                            selected_key=selected_key
                        )
                    )
                label = selected_key
                element_wise_scores.append(ElementWiseScores(label=label, values=temp_hits))
        return [MetricResults(element_wise_scores=element_wise_scores, need_average=True)]


    def _full_match_argument(self, parsed_predicted_arguments, parsed_reference_arguments,
                             selected_key,day_lift=None) -> bool:
        key = selected_key
        possible_day_lift = day_lift
        if 'date' in key:
            try:
                predicted_date_obj = datetime.strptime(parsed_reference_arguments[key], '%Y-%m-%d')
                reference_date_obj = datetime.strptime(parsed_predicted_arguments[key], '%Y-%m-%d')
                if abs((predicted_date_obj - reference_date_obj).days) <= possible_day_lift:
                    return True
                else:
                    return False
            except (ValueError, TypeError):
                return False
        else:
            try:
                return parsed_reference_arguments[self._settings.selected_argument_key] \
                       == parsed_predicted_arguments[self._settings.selected_argument_key]
            except:
                return False


    def _parse_function_with_arguments(self, inputs: InferenceChatDataset | list[list[str]])\
            -> tuple[list, list]:
        functions = []
        arguments = []
        for input_batch in inputs:
            for input in input_batch:
                input_splitted = self._load_json(input)
                if len(input_splitted) == 2:
                    functions.append(input_splitted.get('name',''))
                    arguments.append(input_splitted.get('parameters',''))
                else:
                    functions.append('')
                    arguments.append('')
        return functions, arguments


    @staticmethod
    def _load_json(data):
        try:
            return  json.loads(data)
        except json.JSONDecodeError as e:
            error_position = e.pos
            fixed_json = data[:error_position] + "null," + data[error_position + 1:]
            try:
                return  json.loads(fixed_json)
            except json.JSONDecodeError as e2:
                return {}
