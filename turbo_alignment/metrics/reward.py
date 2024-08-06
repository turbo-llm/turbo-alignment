import torch
from accelerate import Accelerator

from turbo_alignment.common.tf.loaders import load_model, load_tokenizer
from turbo_alignment.dataset.sampling import SamplingRMDataset
from turbo_alignment.generators.rm import RMSamplingGenerator
from turbo_alignment.metrics.metric import Metric
from turbo_alignment.metrics.registry import RewardSettings
from turbo_alignment.settings.datasets.base import DatasetSourceSettings
from turbo_alignment.settings.generators.outputs.chat import AnswerMessage
from turbo_alignment.settings.metric import ElementWiseScores, MetricResults, MetricType


@Metric.register(MetricType.REWARD)
class RewardMetric(Metric):
    def __init__(self, settings: RewardSettings) -> None:
        super().__init__(settings=settings)
        self._settings: RewardSettings = settings
        self.tokenizer = load_tokenizer(
            self._settings.tokenizer_settings,
            self._settings.model_settings,
        )

        self.model = load_model(self._settings.model_settings, self.tokenizer)
        self.model.eval()

    def compute(self, **kwargs) -> list[MetricResults]:
        dataset: SamplingRMDataset = kwargs.get('dataset', None)
        predictions: list[list[str]] = kwargs.get('predictions', None)
        accelerator: Accelerator = kwargs.get('accelerator', None)
        dataset_name: str = kwargs.get('dataset_name', '')

        if dataset is None:
            raise ValueError('dataset should not be None')

        if predictions is None:
            raise ValueError('predictions should not be None')

        self.model = accelerator.prepare_model(self.model, device_placement=True, evaluation_mode=True)
        self.model.to(accelerator.device)

        generator = RMSamplingGenerator(
            model=self.model, tokenizer=self.tokenizer, accelerator=accelerator, micro_batch=self._settings.micro_batch
        )
        answers_per_context = len(predictions[0])

        messages = [record['messages'] for record in dataset.records for _ in range(answers_per_context)]

        answers = [
            AnswerMessage(id=str(ans_idx), content=ans)
            for ctx_answers in predictions
            for ans_idx, ans in enumerate(ctx_answers)
        ]

        rm_input_records = [
            {'id': id, 'dataset_name': '', 'messages': context, 'answers': [answer]}
            for id, (context, answer) in enumerate(zip(messages, answers))
        ]

        chat_settings = dataset.settings
        chat_settings.only_answer_loss = False

        new_dataset = SamplingRMDataset(
            source=DatasetSourceSettings(name='', records_data=rm_input_records, sample_rate=1.0),
            settings=chat_settings,
            tokenizer=self.tokenizer,
        )

        generator_outputs = generator.generate_from_dataset(new_dataset)
        reward_scores = [list(output.rewards.values()) for output in generator_outputs]

        return [
            MetricResults(
                element_wise_scores=[ElementWiseScores(label=dataset_name + '@@' + 'reward', values=reward_scores)],
                need_average=need_average,
            )
            for need_average in self._settings.need_average
        ]


def compute_metrics(eval_preds) -> dict[str, float]:
    logits, labels = eval_preds
    rewards_w, rewards_l = logits[:, 0], logits[:1]

    return {
        'pair_accuracy': labels.mean(),
        'chosen_reward': rewards_w.mean(),
        'rejected_reward': rewards_l.mean(),
        'reward_difference': rewards_w.mean() - rewards_l.mean(),
    }


def pair_comparison_accuracy_score(rewards_w: torch.Tensor, rewards_l: torch.Tensor) -> float:
    pair_scores = [1 if w > l else 0 for w, l in zip(rewards_w, rewards_l)]
    return sum(pair_scores) / len(rewards_w)
