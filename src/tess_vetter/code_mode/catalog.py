"""Deterministic code-mode catalog construction and version hashing."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Literal

CatalogTier = Literal["golden_path", "primitive", "internal"]

TIER_ORDER: tuple[CatalogTier, ...] = ("golden_path", "primitive", "internal")
_TIER_RANK: dict[str, int] = {tier: idx for idx, tier in enumerate(TIER_ORDER)}


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    """Single deterministic catalog entry."""

    id: str
    tier: str
    title: str
    description: str
    tags: tuple[str, ...]
    schema: Any
    schema_fingerprint: str


@dataclass(frozen=True, slots=True)
class CatalogBuildResult:
    """Deterministic catalog build result."""

    entries: tuple[CatalogEntry, ...]
    canonical_lines: tuple[str, ...]
    catalog_version_hash: str


def canonicalize_value(value: Any) -> Any:
    """Canonicalize nested JSON-like values and reject non-finite numbers."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Non-finite numeric value is not supported in schema")
        return value
    if isinstance(value, dict):
        return {
            str(key): canonicalize_value(value[key])
            for key in sorted(value, key=lambda k: str(k))
        }
    if isinstance(value, (list, tuple)):
        return [canonicalize_value(item) for item in value]
    raise TypeError(f"Unsupported schema value type: {type(value)!r}")


def schema_fingerprint(schema: Any) -> str:
    """Return SHA-256 fingerprint of canonicalized schema."""
    canonical_schema = canonicalize_value(schema)
    payload = json.dumps(
        canonical_schema,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _normalize_tags(raw_tags: Any) -> tuple[str, ...]:
    if raw_tags is None:
        return ()
    if isinstance(raw_tags, str):
        tags = [raw_tags]
    elif isinstance(raw_tags, (list, tuple, set)):
        tags = [str(item) for item in raw_tags]
    else:
        raise TypeError("tags must be a string or sequence of strings")

    return tuple(sorted({tag.strip().lower() for tag in tags if tag and tag.strip()}))


def _tier_sort_key(tier: str) -> tuple[int, str]:
    return (_TIER_RANK.get(tier, len(TIER_ORDER)), tier)


def _canonical_line(entry: CatalogEntry) -> str:
    tags = ",".join(entry.tags)
    parts = [
        entry.tier,
        entry.id,
        entry.title,
        entry.description,
        tags,
        entry.schema_fingerprint,
    ]
    return "|".join(parts)


def short_hash(value: str, *, length: int = 12) -> str:
    """Return a compact prefix for a hash string."""
    if length < 1:
        raise ValueError("length must be >= 1")
    return value[:length]


def build_catalog(entries: list[dict[str, Any]]) -> CatalogBuildResult:
    """Build deterministic catalog from input entries."""
    normalized: list[CatalogEntry] = []

    for raw in entries:
        entry_id = str(raw["id"])
        tier = str(raw["tier"])
        title = str(raw.get("title") or raw.get("name") or entry_id)
        description = str(raw.get("description") or "")
        tags = _normalize_tags(raw.get("tags"))

        raw_schema = raw.get("schema", {})
        canonical_schema = canonicalize_value(raw_schema)
        schema_fp = schema_fingerprint(canonical_schema)

        normalized.append(
            CatalogEntry(
                id=entry_id,
                tier=tier,
                title=title,
                description=description,
                tags=tags,
                schema=canonical_schema,
                schema_fingerprint=schema_fp,
            )
        )

    sorted_entries = tuple(sorted(normalized, key=lambda e: (_tier_sort_key(e.tier), e.id)))
    canonical_lines = tuple(_canonical_line(entry) for entry in sorted_entries)
    lines_payload = "\n".join(canonical_lines)
    version_hash = sha256(lines_payload.encode("utf-8")).hexdigest()

    return CatalogBuildResult(
        entries=sorted_entries,
        canonical_lines=canonical_lines,
        catalog_version_hash=version_hash,
    )


__all__ = [
    "CatalogBuildResult",
    "CatalogEntry",
    "CatalogTier",
    "TIER_ORDER",
    "build_catalog",
    "canonicalize_value",
    "schema_fingerprint",
    "short_hash",
]
