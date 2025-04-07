from tests.utils import is_sample_build_from_content
from turbo_alignment.dataset.classification.models import ClassificationDatasetRecord
from turbo_alignment.dataset.pair_preferences.models import PairPreferenceRecord
from turbo_alignment.dataset.registry import DatasetRegistry
from turbo_alignment.settings.datasets.base import DatasetStrategy, DatasetType
from turbo_alignment.settings.datasets.classification import (
    ClassificationDatasetSettings,
)
from turbo_alignment.settings.datasets.pair_preference import (
    PairPreferenceDatasetSettings,
)


def test_classification(tokenizer_llama2, chat_dataset_settings, classification_dataset_source):
    # load dataset and check that samples have required fields

    source, data_dicts = classification_dataset_source

    dataset_cls = DatasetRegistry.by_name(DatasetType.CLASSIFICATION).by_name(DatasetStrategy.TRAIN)

    dataset_settings = ClassificationDatasetSettings(chat_settings=chat_dataset_settings)

    dataset = dataset_cls(tokenizer=tokenizer_llama2, source=source, settings=dataset_settings, seed=42)

    assert len(data_dicts) == len(dataset)

    for data_dict, sample in zip(data_dicts, dataset):
        record = ClassificationDatasetRecord.model_validate(data_dict)

        assert record.label == sample['labels']

        assert is_sample_build_from_content(
            sample['input_ids'], [m.content for m in record.messages], tokenizer_llama2
        )


def test_pair_preferences(tokenizer_llama2, chat_dataset_settings, pair_preferences_dataset_source):
    # load dataset and check that samples have required fields

    source, data_dicts = pair_preferences_dataset_source

    dataset_cls = DatasetRegistry.by_name(DatasetType.PAIR_PREFERENCES).by_name(DatasetStrategy.TRAIN)

    dataset_settings = PairPreferenceDatasetSettings(chat_settings=chat_dataset_settings)
    dataset = dataset_cls(tokenizer=tokenizer_llama2, source=source, settings=dataset_settings, seed=42)

    assert len(data_dicts) == len(dataset)

    for data_dict, sample in zip(data_dicts, dataset):
        record = PairPreferenceRecord.model_validate(data_dict)
        context: list[str] = [c.content for c in record.context]
        contents_w = [*context, record.answer_w.content]
        assert is_sample_build_from_content(sample['inputs_w']['input_ids'], contents_w, tokenizer_llama2)

        contents_l = [*context, record.answer_l.content]
        assert is_sample_build_from_content(sample['inputs_l']['input_ids'], contents_l, tokenizer_llama2)
