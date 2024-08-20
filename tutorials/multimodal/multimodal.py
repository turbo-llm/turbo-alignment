from datasets import load_dataset
import random
import subprocess
import json
from pathlib import Path
from typing import Any


# TMP
def write_jsonl(records: Any, path: Path) -> None:
    with path.open('w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
# TMP


dataset = load_dataset("passing2961/photochat_plus")['train']

data = []

for i, sample in enumerate(dataset):
    data.append({
        'id': i,
        'messages': [
            {
                'role': 'user',
                'type': 'image',
                'content': f'images/00000/{str(i).zfill(9)}.jpg'
            },
            {
                'role': 'bot',
                'type': 'text',
                'content': sample['image_descriptions'][0].strip()
            }
        ]
    })

random.shuffle(data)
threshold = int(0.1 * len(data))
test = data[:threshold]
train = data[threshold:]

write_jsonl(test, Path('val_chat.jsonl'))
write_jsonl(train, Path('train_chat.jsonl'))