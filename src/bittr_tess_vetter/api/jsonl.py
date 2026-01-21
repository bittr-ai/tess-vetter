"""JSONL helpers for bulk enrichment.

This module is intended for batch workflows and CLIs. It avoids pulling in
heavier dependencies and keeps I/O patterns streaming-friendly.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from a JSONL file (streaming)."""
    with path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON at {path}:{line_num}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object at {path}:{line_num}, got {type(obj)}")
            yield obj


def stream_existing_candidate_keys(path: Path) -> set[str]:
    """Collect candidate keys from an existing enriched JSONL output."""
    keys: set[str] = set()
    if not path.exists():
        return keys
    for obj in iter_jsonl(path):
        key = obj.get("candidate_key")
        if isinstance(key, str) and key:
            keys.add(key)
    return keys


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    """Append one JSON object to a JSONL file using a file lock."""
    from filelock import FileLock

    lock = FileLock(str(path.with_suffix(path.suffix + ".lock")), timeout=30)
    with lock, path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), separators=(",", ":")))
        handle.write("\n")


__all__ = ["iter_jsonl", "stream_existing_candidate_keys", "append_jsonl"]

