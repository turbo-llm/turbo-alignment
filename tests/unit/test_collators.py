from tests.utils import is_sample_build_from_content
from turbo_alignment.dataset.kto.collators import KTODataCollator
from turbo_alignment.dataset.registry import DatasetRegistry
from turbo_alignment.settings.datasets.base import DatasetStrategy, DatasetType
from turbo_alignment.settings.datasets.kto import KTODatasetSettings


def test_kto_collator(tokenizer_llama2, chat_dataset_settings, kto_dataset_source):
    tokenizer = tokenizer_llama2
    source, data_dicts = kto_dataset_source

    dataset_cls = DatasetRegistry.by_name(DatasetType.KTO).by_name(DatasetStrategy.TRAIN)

    dataset_settings = KTODatasetSettings(chat_settings=chat_dataset_settings)
    dataset = dataset_cls(tokenizer=tokenizer, source=source, settings=dataset_settings, seed=42)

    batch_size = min(len(dataset), 8)
    examples = list(dataset)[:batch_size]

    collator = KTODataCollator(tokenizer=tokenizer)
    batch = collator(examples)

    ignore_index = -100
    for answer_labels, is_desirable, raw_data in zip(batch['labels'], batch['is_desirable'], data_dicts):
        answer = raw_data['answer']['content']
        answer_tokens = answer_labels[answer_labels != ignore_index]
        assert is_sample_build_from_content(answer_tokens, [answer], tokenizer)
        assert raw_data['is_desirable'] == is_desirable
