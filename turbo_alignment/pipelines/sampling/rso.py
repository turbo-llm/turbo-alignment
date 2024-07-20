from copy import deepcopy

import numpy as np

from turbo_alignment.dataset.sampling.models import SamplingDatasetRecord
from turbo_alignment.dataset.sampling.rm import SamplingRMDataset
from turbo_alignment.pipelines.sampling.rm import BaseSamplingStrategyWithRM
from turbo_alignment.settings.pipelines.sampling import RSOSamplingSettings


class RSOSamplingStrategy(BaseSamplingStrategyWithRM[RSOSamplingSettings]):
    @staticmethod
    def __sample_for_single_record(
        experiment_settings: RSOSamplingSettings,
        record: SamplingDatasetRecord,
        candidates: dict[str, float],
    ) -> SamplingDatasetRecord:
        accepted: list[str] = []

        while len(accepted) < experiment_settings.N and candidates:
            max_reward = max(candidates.values())

            for c, r in candidates.items():
                u = np.random.uniform()
                if u >= np.exp((r - max_reward) / experiment_settings.beta):
                    continue

                accepted.append(c)

                if len(accepted) == experiment_settings.N:
                    break

            to_remove = []
            for c in candidates:
                if c not in accepted:
                    to_remove.append(c)
            for c in to_remove:
                candidates.pop(c)

        sampled_record = deepcopy(record)
        sampled_record.rewards = candidates

        answers = []
        for answer in sampled_record.answers:
            if answer.id in candidates:
                answers.append(answer)
        sampled_record.answers = answers
        return sampled_record

    @staticmethod
    def _sample(
        experiment_settings: RSOSamplingSettings,
        dataset: SamplingRMDataset,
        rewards: dict[str, dict[str, float]],
    ) -> list[SamplingDatasetRecord]:
        sampled_records: list[SamplingDatasetRecord] = []

        for record_id, record in dataset.original_records_map.items():
            sampled_records.append(
                RSOSamplingStrategy.__sample_for_single_record(
                    experiment_settings=experiment_settings,
                    record=record,
                    candidates=rewards[record_id],
                )
            )

        return sampled_records
