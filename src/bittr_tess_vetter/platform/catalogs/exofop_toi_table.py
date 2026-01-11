from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

EXOFOP_TOI_URL = "https://exofop.ipac.caltech.edu/tess/download_toi.php"
CONNECT_TIMEOUT_SECONDS = 10.0
READ_TIMEOUT_SECONDS = 45.0
MAX_RETRIES = 3


def _default_cache_dir() -> Path:
    root = os.getenv("BITTR_TESS_VETTER_CACHE_ROOT") or os.getenv("ASTRO_ARC_CACHE_ROOT")
    if root:
        return Path(root)
    return Path.cwd() / ".bittr-tess-vetter" / "cache"


def _toi_cache_path(cache_dir: Path) -> Path:
    return Path(cache_dir) / "exofop" / "toi_table.pipe"


def _read_disk_cache(path: Path, *, cache_ttl_seconds: int) -> str | None:
    if cache_ttl_seconds <= 0:
        return None
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    except Exception:
        return None
    age = time.time() - float(stat.st_mtime)
    if age > float(cache_ttl_seconds):
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def _write_disk_cache(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _normalize_header(col: str) -> str:
    col = (col or "").strip().lower()
    col = col.replace("%", "pct")
    # Keep this conservative; we want stable keys without losing readability.
    for ch in ["(", ")", "[", "]", "{", "}", ",", ":", ";", "/"]:
        col = col.replace(ch, " ")
    col = col.replace("-", " ")
    col = "_".join(x for x in col.split() if x)
    return col


def _parse_pipe_table(text: str) -> tuple[list[str], list[dict[str, str]]]:
    lines = (text or "").strip().splitlines()
    if not lines:
        return [], []
    header_raw = [h.strip() for h in lines[0].split("|")]
    header_norm = [_normalize_header(h) for h in header_raw]
    rows: list[dict[str, str]] = []
    for line in lines[1:]:
        cols = [c.strip() for c in line.split("|")]
        if len(cols) < 2:
            continue
        row: dict[str, str] = {}
        for k, v in zip(header_norm, cols, strict=False):
            if not k:
                continue
            row[k] = v
        rows.append(row)
    return header_norm, rows


@dataclass(frozen=True)
class ExoFOPToiTable:
    fetched_at_unix: float
    headers: list[str]
    rows: list[dict[str, str]]

    def entries_for_tic(self, tic_id: int) -> list[dict[str, str]]:
        target = str(int(tic_id))
        # Common header variants: "tic_id", "tic", "ticid"
        tic_keys = ["tic_id", "tic", "ticid"]
        out: list[dict[str, str]] = []
        for row in self.rows:
            for k in tic_keys:
                if row.get(k) == target:
                    out.append(row)
                    break
        return out


_CACHE: ExoFOPToiTable | None = None


def fetch_exofop_toi_table(
    *,
    cache_ttl_seconds: int = 6 * 3600,
    disk_cache_dir: str | Path | None = None,
) -> ExoFOPToiTable:
    """Fetch and parse the ExoFOP TOI table (pipe-delimited).

    This is the fastest way to avoid duplicating TFOP work (dispositions + comments).
    Caches the full table:
    - in-process (fast for repeated calls within one server process)
    - on-disk (persists across MCP process restarts)
    """
    global _CACHE
    now = time.time()
    if _CACHE is not None and cache_ttl_seconds > 0:
        if (now - _CACHE.fetched_at_unix) <= float(cache_ttl_seconds):
            return _CACHE

    cache_dir = Path(disk_cache_dir) if disk_cache_dir is not None else _default_cache_dir()
    cache_path = _toi_cache_path(cache_dir)
    cached_text = _read_disk_cache(cache_path, cache_ttl_seconds=int(cache_ttl_seconds))
    if cached_text:
        headers, rows = _parse_pipe_table(cached_text)
        table = ExoFOPToiTable(fetched_at_unix=float(cache_path.stat().st_mtime), headers=headers, rows=rows)
        _CACHE = table
        return table

    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(
                EXOFOP_TOI_URL,
                params={"sort": "toi", "output": "pipe"},
                timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
            )
            response.raise_for_status()
            break
        except Exception as e:
            last_exc = e
            if attempt >= MAX_RETRIES - 1:
                raise
            # Simple backoff: 1s, 2s
            time.sleep(2**attempt)
    else:
        # Defensive; should never happen.
        raise last_exc or RuntimeError("ExoFOP fetch failed")

    # Persist to disk so MCP subprocess-based clients don't redownload on every call.
    try:
        _write_disk_cache(cache_path, response.text)
    except Exception:
        # Cache failures should never block analysis.
        pass

    headers, rows = _parse_pipe_table(response.text)
    table = ExoFOPToiTable(fetched_at_unix=now, headers=headers, rows=rows)
    _CACHE = table
    return table


def exofop_entries_for_tic(
    *,
    tic_id: int,
    cache_ttl_seconds: int = 6 * 3600,
) -> list[dict[str, Any]]:
    """Return ExoFOP TOI table entries for a TIC ID.

    Returns raw rows (string values) keyed by normalized header.
    """
    table = fetch_exofop_toi_table(cache_ttl_seconds=int(cache_ttl_seconds))
    return [dict(row) for row in table.entries_for_tic(int(tic_id))]
