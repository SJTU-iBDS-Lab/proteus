from __future__ import annotations

import json
from pathlib import Path


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_jsonl_records(path: Path, records: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            _ = handle.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )
    return path


def write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def write_markdown(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_text(
        content if content.endswith("\n") else content + "\n", encoding="utf-8"
    )
    return path
