"""Tiny JSON / JSONL persistence for the tracking features. Everything lives under data/ (git-ignored).

Set EQR_DATA to point somewhere else (the tests do). Writes are atomic so a crash never leaves half a file.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def data_dir() -> Path:
    return Path(os.environ.get("EQR_DATA", "data"))


def read_json(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")
