from tests.utils import is_sample_build_from_content
from turbo_alignment.dataset.classification.models import ClassificationDatasetRecord
from turbo_alignment.dataset.pair_preferences.models import PairPreferenceRecord
from turbo_alignment.dataset.registry import DatasetRegistry
from turbo_alignment.settings.datasets.base import DatasetStrategy, DatasetType
from turbo_alignment.settings.datasets.classification import (
    ClassificationDatasetSettings,
)
from turbo_alignment.settings.datasets.ddpo import DDPODatasetSettings
from turbo_alignment.settings.datasets.pair_preference import (
    PairPreferenceDatasetSettings,
)


def test_classification(tokenizer_llama2, chat_dataset_settings, classification_dataset_source):
    # load dataset and check that samples have required fields

    source, data_dicts = classification_dataset_source

    dataset_cls = DatasetRegistry.by_name(DatasetType.CLASSIFICATION).by_name(DatasetStrategy.TRAIN)

    dataset_settings = ClassificationDatasetSettings(chat_settings=chat_dataset_settings)

    dataset = dataset_cls(tokenizer=tokenizer_llama2, source=source, settings=dataset_settings)

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
    dataset = dataset_cls(tokenizer=tokenizer_llama2, source=source, settings=dataset_settings)

    assert len(data_dicts) == len(dataset)

    for data_dict, sample in zip(data_dicts, dataset):
        record = PairPreferenceRecord.model_validate(data_dict)
        context: list[str] = [c.content for c in record.context]
        contents_w = [*context, record.answer_w.content]
        assert is_sample_build_from_content(sample['inputs_w']['input_ids'], contents_w, tokenizer_llama2)

        contents_l = [*context, record.answer_l.content]
        assert is_sample_build_from_content(sample['inputs_l']['input_ids'], contents_l, tokenizer_llama2)


def test_ddpo(tokenizer_llama2, tokenizer_gptj, chat_dataset_settings, pair_preferences_dataset_source):
    sft_tokenizer = tokenizer_llama2
    rm_tokenizer = tokenizer_gptj
    # load dataset and check that samples have required fields

    source, data_dicts = pair_preferences_dataset_source

    dataset_cls = DatasetRegistry.by_name(DatasetType.DDPO).by_name(DatasetStrategy.TRAIN)

    pair_preferences_dataset_settings = PairPreferenceDatasetSettings(chat_settings=chat_dataset_settings)
    dataset_settings = DDPODatasetSettings(
        chat_settings=chat_dataset_settings, pair_preferences=pair_preferences_dataset_settings
    )
    dataset = dataset_cls(
        chat_tokenizer=sft_tokenizer, rm_tokenizer=rm_tokenizer, source=source, settings=dataset_settings
    )

    assert len(data_dicts) == len(dataset)

    for data_dict, sample in zip(data_dicts, dataset):
        record = PairPreferenceRecord.model_validate(data_dict)
        context: list[str] = [c.content for c in record.context]
        contents_w = [*context, record.answer_w.content]
        assert is_sample_build_from_content(sample['sft_inputs_w']['input_ids'], contents_w, sft_tokenizer)
        assert is_sample_build_from_content(sample['rm_inputs_w']['input_ids'], contents_w, rm_tokenizer)

        contents_l = [*context, record.answer_l.content]
        assert is_sample_build_from_content(sample['sft_inputs_l']['input_ids'], contents_l, sft_tokenizer)
        assert is_sample_build_from_content(sample['rm_inputs_l']['input_ids'], contents_l, rm_tokenizer)
