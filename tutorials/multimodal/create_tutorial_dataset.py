import json
import random
import subprocess
from pathlib import Path
from typing import Any

from datasets import load_dataset

from turbo_alignment.common.data.io import write_jsonl
from turbo_alignment.dataset.chat.models import ChatMessageRole
from turbo_alignment.dataset.multimodal.models import (
    MultimodalChatMessage,
    MultimodalDatasetRecord,
    MultimodalImageMessage,
    MultimodalTextMessage,
)
from turbo_alignment.settings.modality import Modality


def convert_to_multimodal_record(row):
    return MultimodalDatasetRecord(
        id=row['id'],
        messages=[
            MultimodalImageMessage(role=ChatMessageRole.USER, content=f"images/00000/{str(row['id']).zfill(9)}.jpg"),
            MultimodalTextMessage(role=ChatMessageRole.BOT, content=row['image_descriptions'][0].strip()),
        ],
    ).dict()


if __name__ == '__main__':
    dataset = load_dataset('passing2961/photochat_plus')['train']
    dataset = dataset.add_column('id', range(len(dataset)))
    dataset = dataset.train_test_split(test_size=0.1)

    dataset['train_multimodal_records'] = dataset['train'].map(
        convert_to_multimodal_record, remove_columns=dataset['train'].column_names
    )
    dataset['val_multimodal_records'] = dataset['test'].map(
        convert_to_multimodal_record, remove_columns=dataset['test'].column_names
    )

    write_jsonl([item for item in dataset['train_multimodal_records']], Path('train_multimodal.jsonl'))
    write_jsonl([item for item in dataset['val_multimodal_records']], Path('val_multimodal.jsonl'))
