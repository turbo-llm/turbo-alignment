from typing import Generic, Type, TypeVar

from transformers import PreTrainedTokenizerBase

from turbo_alignment.common.logging import get_project_logger
from turbo_alignment.dataset.base import BaseDataset
from turbo_alignment.dataset.registry import DatasetRegistry
from turbo_alignment.settings.datasets.base import DatasetStrategy, MultiDatasetSettings

logger = get_project_logger()


DatasetType = TypeVar('DatasetType', bound=BaseDataset)


class DatasetLoader(Generic[DatasetType]):
    def __init__(self, cls: Type[DatasetType] | None = None):
        self._dataset_cls = cls

    def load_datasets(
        self,
        multi_dataset_settings: MultiDatasetSettings,
        tokenizer: PreTrainedTokenizerBase,
        strategy: DatasetStrategy,
    ) -> list[DatasetType]:
        logger.info(
            f'Loading dataset {multi_dataset_settings.dataset_type} with settings:\n{multi_dataset_settings.dict()}'
        )

        datasets: list[DatasetType] = []
        for source in multi_dataset_settings.sources:
            # Determining what type of dataset is in the nested registry by field 'strategy'

            dataset = DatasetRegistry.by_name(multi_dataset_settings.dataset_type).by_name(strategy)(
                tokenizer=tokenizer,
                source=source,
                settings=multi_dataset_settings,
            )

            if self._dataset_cls is not None:
                assert isinstance(dataset, self._dataset_cls)

            datasets.append(dataset)

        return datasets
