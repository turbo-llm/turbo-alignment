from datasets import load_dataset
from turbo_alignment.dataset.kto.models import KTODatasetRecord
import random

from turbo_alignment.common.data.io import write_jsonl
from pathlib import Path

import subprocess


def convert_to_kto_record(row: dict[str, str | int]) -> KTODatasetRecord:
    assert row['messages'][:-1] == row['chosen'][:-1]
    assert row['messages'][:-1] == row['rejected'][:-1]

    for msg in row['messages']:
        if msg['role'] == 'assistant':
            msg['role'] = 'bot'
    
    row['chosen'][-1]['role'] = 'bot'
    row['rejected'][-1]['role'] = 'bot'

    if random.random() <= 0.5:
        return KTODatasetRecord(
            id=row['prompt_id'],
            context=row['messages'][:-1],
            answer=row['chosen'][-1],
            is_desirable=True,
        ).dict()
    
    return KTODatasetRecord(
            id=row['prompt_id'],
            context=row['messages'][:-1],
            answer=row['rejected'][-1],
            is_desirable=False,
        ).dict()


if __name__ == "__main__":
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized")

    ds['train_kto_records'] = ds['train_prefs'].map(convert_to_kto_record, remove_columns=ds['train_prefs'].column_names)
    ds['val_kto_records'] = ds['test_prefs'].map(convert_to_kto_record, remove_columns=ds['test_prefs'].column_names)


    write_jsonl([item for item in ds['train_kto_records']], Path('train_kto.jsonl'))
    write_jsonl([item for item in ds['val_kto_records']], Path('val_kto.jsonl'))


    launch_code = "poetry run python -m turbo_alignment train_kto --experiment_settings_path turbo-alignment/tutorials/kto/kto.json"
    subprocess.run(launch_code, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
