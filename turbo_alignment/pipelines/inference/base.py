from abc import abstractmethod
from typing import Generator, Generic, TypeVar

from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.utils.operations import gather_object
from pydantic import BaseModel
from transformers import PreTrainedTokenizerBase

from turbo_alignment.common.data.io import write_json, write_jsonl
from turbo_alignment.common.logging import get_project_logger
from turbo_alignment.dataset.base import BaseDataset
from turbo_alignment.dataset.loader import DatasetLoader
from turbo_alignment.generators.base import BaseGenerator
from turbo_alignment.pipelines.base import BaseStrategy
from turbo_alignment.settings.datasets.base import DatasetStrategy
from turbo_alignment.settings.generators.outputs.base import BaseInferenceOutput
from turbo_alignment.settings.pipelines.inference import InferenceExperimentSettings

logger = get_project_logger()

InferenceOutputRecordT = TypeVar('InferenceOutputRecordT', bound=BaseInferenceOutput)
InferenceExperimentSettingsT = TypeVar('InferenceExperimentSettingsT', bound=InferenceExperimentSettings)


class BaseInferenceStrategy(BaseStrategy, Generic[InferenceExperimentSettingsT]):
    @abstractmethod
    def _get_single_inference_settings(
        self,
        experiment_settings: InferenceExperimentSettingsT,
        accelerator: Accelerator,
    ) -> Generator[tuple[PreTrainedTokenizerBase, BaseGenerator, str, dict], None, None]:
        ...

    def run(self, experiment_settings: InferenceExperimentSettingsT) -> None:
        accelerator = Accelerator()
        set_seed(seed=0, device_specific=False)
        experiment_settings.save_path.mkdir(parents=True, exist_ok=True)

        report = {}
        for tokenizer, generator, filename, parameters_to_save in self._get_single_inference_settings(
            experiment_settings, accelerator
        ):
            datasets = DatasetLoader[BaseDataset](BaseDataset).load_datasets(
                experiment_settings.dataset_settings,
                tokenizer=tokenizer,
                strategy=DatasetStrategy.INFERENCE,
            )
            generations_output: list[BaseModel] = sum(
                [gather_object(generator.generate_from_dataset(dataset)) for dataset in datasets], []
            )

            write_jsonl(
                [out.dict(exclude_none=True) for out in generations_output], experiment_settings.save_path / filename
            )

            report[filename] = parameters_to_save

        write_json(report, experiment_settings.save_path / 'info.json')
