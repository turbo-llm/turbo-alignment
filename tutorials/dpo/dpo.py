from datasets import load_dataset
from turbo_alignment.dataset.pair_preferences.models import PairPreferenceRecord
from turbo_alignment.dataset.chat.chat import ChatDatasetRecord

from turbo_alignment.common.data.io import write_jsonl
from pathlib import Path

import subprocess


def convert_to_preference_record(row: dict[str, str | int]) -> PairPreferenceRecord:
    assert row['messages'][:-1] == row['chosen'][:-1]
    assert row['messages'][:-1] == row['rejected'][:-1]

    for msg in row['messages']:
        if msg['role'] == 'assistant':
            msg['role'] = 'bot'
    
    row['chosen'][-1]['role'] = 'bot'
    row['rejected'][-1]['role'] = 'bot'

    return PairPreferenceRecord(
        id=row['prompt_id'],
        context=row['messages'][:-1],
        answer_w=row['chosen'][-1],
        answer_l=row['rejected'][-1],
    ).dict()


def convert_to_chat_record(row: dict[str, str | int]) -> ChatDatasetRecord:
    for msg in row['messages']:
        if msg['role'] == 'assistant':
            msg['role'] = 'bot'

    return ChatDatasetRecord(
        id=row['prompt_id'],
        messages=row['messages'][:-1],
    ).dict()


if __name__ == "__main__":
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized")

    ds['train_pair_preference_records'] = ds['train_prefs'].map(convert_to_preference_record, remove_columns=ds['train_prefs'].column_names)
    ds['val_pair_preference_records'] = ds['test_prefs'].map(convert_to_preference_record, remove_columns=ds['test_prefs'].column_names)
    ds['val_chat_records'] = ds['test_prefs'].map(convert_to_chat_record, remove_columns=ds['test_prefs'].column_names)


    write_jsonl([item for item in ds['train_pair_preference_records']], Path('train_preferences.jsonl'))
    write_jsonl([item for item in ds['val_pair_preference_records']], Path('val_preferences.jsonl'))
    write_jsonl([item for item in ds['val_chat_records']], Path('val_chat.jsonl'))


    launch_code = "poetry run python -m turbo_alignment train_dpo --experiment_settings_path turbo-alignment/tutorials/dpo/dpo.json"
    subprocess.run(launch_code, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
