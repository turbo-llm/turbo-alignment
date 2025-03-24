import json

from pytest import fixture
from transformers import AutoTokenizer

from turbo_alignment.dataset.registry import DatasetRegistry
from turbo_alignment.settings.datasets.base import (
    DatasetSourceSettings,
    DatasetStrategy,
    DatasetType,
)
from turbo_alignment.settings.datasets.chat import (
    ChatDatasetSettings,
    ChatPromptTemplate,
)
from turbo_alignment.settings.datasets.pair_preference import (
    PairPreferenceDatasetSettings,
)


@fixture(scope='session')
def tokenizer_llama2():
    tokenizer_path = 'tests/fixtures/models/llama2_classification/tokenizer'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return tokenizer


@fixture(scope='session')
def tokenizer_gptj():
    tokenizer_path = 'tests/fixtures/models/gptj_tiny_for_seq_cls'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return tokenizer


@fixture(scope='session')
def chat_dataset_settings():
    chat_dataset_settings = ChatDatasetSettings(
        prompt_template=ChatPromptTemplate(
            prefix_template='<S>',
            suffix_template='</S>',
            role_tag_mapping={'user': 'USER', 'bot': 'BOT', 'system': 'SYSTEM'},
        ),
        max_tokens_count=8256,
    )
    return chat_dataset_settings


@fixture(scope='session')
def classification_dataset_path() -> str:
    return 'tests/fixtures/datasets/classification/train_classification.jsonl'


@fixture(scope='session')
def pair_preferences_dataset_path() -> str:
    return 'tests/fixtures/datasets/rm/train_preferences.jsonl'


@fixture(scope='session')
def kto_dataset_path() -> str:
    return 'tests/fixtures/datasets/rm/train_kto.jsonl'


def load_dataset_source(dataset_path: str) -> tuple[DatasetSourceSettings, list[dict]]:
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data_dicts = [json.loads(line) for line in f]

    source = DatasetSourceSettings(name='dataset_for_test', records_path=dataset_path, sample_rate=1)

    return source, data_dicts


@fixture(scope='session')
def classification_dataset_source(classification_dataset_path) -> tuple[DatasetSourceSettings, list[dict]]:
    return load_dataset_source(classification_dataset_path)


@fixture(scope='session')
def pair_preferences_dataset_source(pair_preferences_dataset_path) -> tuple[DatasetSourceSettings, list[dict]]:
    return load_dataset_source(pair_preferences_dataset_path)


@fixture(scope='session')
def kto_dataset_source(kto_dataset_path) -> tuple[DatasetSourceSettings, list[dict]]:
    return load_dataset_source(kto_dataset_path)


@fixture(scope='session')
def dpo_dataset(pair_preferences_dataset_source, tokenizer_llama2, chat_dataset_settings):
    source, _ = pair_preferences_dataset_source

    dataset_cls = DatasetRegistry.by_name(DatasetType.PAIR_PREFERENCES).by_name(DatasetStrategy.TRAIN)

    dataset_settings = PairPreferenceDatasetSettings(chat_settings=chat_dataset_settings)
    dataset = dataset_cls(tokenizer=tokenizer_llama2, source=source, settings=dataset_settings, seed=42)

    return dataset
