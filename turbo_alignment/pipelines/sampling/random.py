import random

from turbo_alignment.common.data.io import read_jsonl
from turbo_alignment.dataset.sampling.models import SamplingDatasetRecord
from turbo_alignment.pipelines.sampling.base import BaseSamplingStrategy
from turbo_alignment.settings.pipelines.sampling import RandomSamplingSettings


class RandomSamplingStrategy(BaseSamplingStrategy):
    def sample(self, experiment_settings: RandomSamplingSettings) -> list[SamplingDatasetRecord]:
        if experiment_settings.dataset_settings.sources[0].records_path:
            records = [
                SamplingDatasetRecord(**record)
                for record in read_jsonl(experiment_settings.dataset_settings.sources[0].records_path)
            ]
        else:
            raise ValueError('records_path should be not None!')

        for record in records:
            record.answers = random.sample(record.answers, min(experiment_settings.N, len(record.answers)))

        return records
