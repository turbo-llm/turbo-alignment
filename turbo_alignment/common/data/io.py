import json
from pathlib import Path
from typing import Any


def read_json(path: Path):
    with path.open(encoding='utf8') as f:
        return json.load(f)


def write_json(data: Any, path: Path) -> None:
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4, default=str)


def read_jsonl(path: Path):
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def write_jsonl(records: Any, path: Path) -> None:
    with path.open('w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def read_txt(path: Path) -> str:
    with path.open(encoding='utf-8') as f:
        return f.read()


def write_string_to_file(string: str, path: Path) -> None:
    with path.open('w', encoding='utf-8') as f:
        f.write(string)


def write_labels(labels: list[str], path: Path) -> None:
    label_str = '\n'.join(labels) + '\n'
    write_string_to_file(label_str, path)


def create_placeholder_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_string_to_file('Placeholder', path)


def read_text_file(path: Path) -> str:
    with path.open(encoding='utf8') as f:
        return f.read()
