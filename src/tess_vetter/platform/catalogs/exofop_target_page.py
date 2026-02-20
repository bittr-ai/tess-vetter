from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

EXOFOP_TARGET_URL = "https://exofop.ipac.caltech.edu/tess/target.php"
REQUEST_TIMEOUT_SECONDS = 20.0


@dataclass(frozen=True)
class ExoFOPTargetSummary:
    tic_id: int
    fetched_at_unix: float
    url: str
    grid_badges: dict[str, int]
    followup_counts: dict[str, int]
    flags: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tic_id": int(self.tic_id),
            "url": self.url,
            "fetched_at_unix": float(self.fetched_at_unix),
            "grid_badges": dict(self.grid_badges),
            "followup_counts": dict(self.followup_counts),
            "flags": list(self.flags),
        }


_CACHE: dict[int, ExoFOPTargetSummary] = {}


_GRID_BADGE_RE = re.compile(
    r'<div\s+class="grid_header">\s*([^<]+?)\s*<span\s+class="grid_badge">\s*(\d+)\s*</span>',
    re.IGNORECASE,
)


def _parse_grid_badges(html: str) -> dict[str, int]:
    badges: dict[str, int] = {}
    for title, count_s in _GRID_BADGE_RE.findall(html or ""):
        title_clean = " ".join((title or "").split()).strip()
        if not title_clean:
            continue
        try:
            badges[title_clean] = int(count_s)
        except Exception:
            continue
    return badges


def _extract_followup_counts(badges: dict[str, int]) -> dict[str, int]:
    # ExoFOP's naming is fairly stable; keep a small mapping to what agents care about.
    mapping = {
        "Imaging Observations": "imaging",
        "Spectroscopy Observations": "spectroscopy",
        "Time Series Observations": "time_series",
        "Files": "files",
    }
    out: dict[str, int] = {}
    for k, v in mapping.items():
        if k in badges:
            out[v] = int(badges[k])
    return out


def _default_cache_dir() -> Path:
    root = os.getenv("BITTR_TESS_VETTER_CACHE_ROOT") or os.getenv("ASTRO_ARC_CACHE_ROOT")
    if root:
        return Path(root)
    return Path.cwd() / ".tess-vetter" / "cache"


def _target_cache_path(cache_dir: Path, tic_id: int) -> Path:
    return Path(cache_dir) / "exofop" / "target_pages" / f"{int(tic_id)}.json"


def _read_disk_cache(path: Path, *, cache_ttl_seconds: int) -> ExoFOPTargetSummary | None:
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
        raw = json.loads(path.read_text(encoding="utf-8"))
        summary = ExoFOPTargetSummary(
            tic_id=int(raw["tic_id"]),
            fetched_at_unix=float(raw.get("fetched_at_unix") or stat.st_mtime),
            url=str(raw.get("url") or ""),
            grid_badges={str(k): int(v) for k, v in dict(raw.get("grid_badges") or {}).items()},
            followup_counts={
                str(k): int(v) for k, v in dict(raw.get("followup_counts") or {}).items()
            },
            flags=[str(x) for x in list(raw.get("flags") or [])],
        )
        # Treat "no badges" as a poisoned/invalid cache entry: ExoFOP pages always have
        # grid headers with numeric badges (often 0), so an empty badge dict strongly
        # implies a transient fetch/parse failure that shouldn't be cached.
        if not summary.grid_badges:
            return None
        return summary
    except Exception:
        return None


def _write_disk_cache(path: Path, summary: ExoFOPTargetSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(summary.to_dict(), sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def fetch_exofop_target_summary(
    *,
    tic_id: int,
    cache_ttl_seconds: int = 6 * 3600,
    disk_cache_dir: str | Path | None = None,
) -> ExoFOPTargetSummary:
    """Fetch ExoFOP target page and extract follow-up "already done" signals.

    This is intentionally cheap: we parse only the badge counts, not the full tables.
    """
    tic_id_i = int(tic_id)
    now = time.time()
    if cache_ttl_seconds > 0 and tic_id_i in _CACHE:
        existing = _CACHE[tic_id_i]
        if existing.grid_badges and (now - existing.fetched_at_unix) <= float(cache_ttl_seconds):
            return existing

    cache_dir = Path(disk_cache_dir) if disk_cache_dir is not None else _default_cache_dir()
    cache_path = _target_cache_path(cache_dir, tic_id_i)
    cached = _read_disk_cache(cache_path, cache_ttl_seconds=int(cache_ttl_seconds))
    if cached is not None:
        _CACHE[tic_id_i] = cached
        return cached

    url = f"{EXOFOP_TARGET_URL}?id={tic_id_i}"
    flags: list[str] = []
    response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    html = response.text
    badges = _parse_grid_badges(html)
    if not badges:
        flags.append("exofop_parse_no_badges")
    followup_counts = _extract_followup_counts(badges)
    if not followup_counts:
        flags.append("exofop_parse_no_followup_counts")

    summary = ExoFOPTargetSummary(
        tic_id=tic_id_i,
        fetched_at_unix=now,
        url=url,
        grid_badges=badges,
        followup_counts=followup_counts,
        flags=flags,
    )
    # Avoid caching parse failures (prevents "no activity" false negatives).
    if cache_ttl_seconds > 0 and summary.grid_badges:
        _CACHE[tic_id_i] = summary
    try:
        if cache_ttl_seconds > 0 and summary.grid_badges:
            _write_disk_cache(cache_path, summary)
    except Exception:
        pass
    return summary
