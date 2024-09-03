from abc import abstractmethod
from typing import Generic, TypeVar

from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.utils.operations import gather_object
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from turbo_alignment.common.tf.loaders.model.model import load_model
from turbo_alignment.common.tf.loaders.tokenizer import load_tokenizer
from turbo_alignment.dataset.sampling.models import SamplingDatasetRecord
from turbo_alignment.dataset.sampling.rm import SamplingRMDataset
from turbo_alignment.generators.rm import RMSamplingGenerator
from turbo_alignment.pipelines.sampling.base import BaseSamplingStrategy
from turbo_alignment.settings.generators.outputs.rm import RMSamplingInferenceOutput
from turbo_alignment.settings.pipelines.sampling import (
    BaseSamplingWithRMSettings,
    SamplingWithRMSettings,
)

SamplingSettingsWithRMT = TypeVar('SamplingSettingsWithRMT', bound=BaseSamplingWithRMSettings)


class BaseSamplingStrategyWithRM(BaseSamplingStrategy[SamplingSettingsWithRMT], Generic[SamplingSettingsWithRMT]):
    @staticmethod
    def _get_rewards(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        dataset: SamplingRMDataset,
        experiment_settings: SamplingSettingsWithRMT,
        accelerator: Accelerator,
    ) -> dict[str, dict[str, float]]:
        generator = RMSamplingGenerator(
            model=model,
            tokenizer=tokenizer,
            accelerator=accelerator,
            batch=experiment_settings.rm_batch_size,
            micro_batch=experiment_settings.rm_batch_size,
        )
        outputs: list[RMSamplingInferenceOutput] = gather_object(generator.generate_from_dataset(dataset))[
            : len(dataset)
        ]
        return {out.id: out.rewards for out in outputs}

    @staticmethod
    @abstractmethod
    def _sample(
        experiment_settings: SamplingSettingsWithRMT,
        dataset: SamplingRMDataset,
        rewards: dict[str, dict[str, float]],
    ) -> list[SamplingDatasetRecord]:
        ...

    def sample(self, experiment_settings: SamplingSettingsWithRMT) -> list[SamplingDatasetRecord]:
        accelerator = Accelerator()
        set_seed(seed=0, device_specific=False)

        tokenizer = load_tokenizer(
            experiment_settings.tokenizer_settings,
            experiment_settings.rm,
        )

        model = load_model(experiment_settings.rm, tokenizer)
        if accelerator is not None:
            model = accelerator.prepare_model(model, device_placement=True, evaluation_mode=True)

        model.eval()

        dataset = SamplingRMDataset(
            source=experiment_settings.dataset_settings.sources[0],
            settings=experiment_settings.dataset_settings.chat_dataset,
            tokenizer=tokenizer,
        )

        rewards = self._get_rewards(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            experiment_settings=experiment_settings,
            accelerator=accelerator,
        )

        return self._sample(experiment_settings, dataset, rewards)


class SamplingStrategyWithRM(BaseSamplingStrategyWithRM[SamplingWithRMSettings]):
    @staticmethod
    def _sample(
        experiment_settings: SamplingWithRMSettings,
        dataset: SamplingRMDataset,
        rewards: dict[str, dict[str, float]],
    ) -> list[SamplingDatasetRecord]:
        records = []

        for record_id, record in dataset.original_records_map.items():
            record.rewards = rewards[record_id]
            records.append(record)

        return records
