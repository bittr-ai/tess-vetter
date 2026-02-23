"""Helpers for standardized ``reference_sources.v1`` payload handling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tess_vetter.cli.common_cli import EXIT_INPUT_ERROR, BtvCliError, load_json_file

REFERENCE_SOURCES_SCHEMA_VERSION = "reference_sources.v1"


def _finite_float(value: Any, *, field_name: str, source_label: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise BtvCliError(
            f"{source_label} field '{field_name}' must be a finite number",
            exit_code=EXIT_INPUT_ERROR,
        ) from exc
    if not np.isfinite(out):
        raise BtvCliError(
            f"{source_label} field '{field_name}' must be a finite number",
            exit_code=EXIT_INPUT_ERROR,
        )
    return out


def _normalize_reference_source(entry: dict[str, Any], *, idx: int, source_label: str) -> dict[str, Any]:
    label = f"{source_label} reference_sources[{int(idx)}]"
    name_raw = entry.get("name")
    if not isinstance(name_raw, str) or not name_raw.strip():
        raise BtvCliError(
            f"{label} must include non-empty string field 'name'",
            exit_code=EXIT_INPUT_ERROR,
        )

    normalized: dict[str, Any] = {
        "name": name_raw.strip(),
        "ra": _finite_float(entry.get("ra"), field_name="ra", source_label=label),
        "dec": _finite_float(entry.get("dec"), field_name="dec", source_label=label),
    }

    source_id = entry.get("source_id")
    if isinstance(source_id, str) and source_id.strip():
        normalized["source_id"] = source_id.strip()

    meta = entry.get("meta")
    if isinstance(meta, dict):
        normalized["meta"] = dict(meta)

    return normalized


def normalize_reference_sources_payload(payload: dict[str, Any], *, source_label: str) -> list[dict[str, Any]]:
    """Validate and normalize a ``reference_sources.v1`` payload."""
    schema_version = payload.get("schema_version")
    if schema_version != REFERENCE_SOURCES_SCHEMA_VERSION:
        raise BtvCliError(
            f"{source_label} must declare schema_version='{REFERENCE_SOURCES_SCHEMA_VERSION}'",
            exit_code=EXIT_INPUT_ERROR,
        )

    raw_sources = payload.get("reference_sources")
    if not isinstance(raw_sources, list) or len(raw_sources) == 0:
        raise BtvCliError(
            f"{source_label} must include non-empty list field 'reference_sources'",
            exit_code=EXIT_INPUT_ERROR,
        )

    normalized: list[dict[str, Any]] = []
    for idx, entry in enumerate(raw_sources):
        if not isinstance(entry, dict):
            raise BtvCliError(
                f"{source_label} reference_sources[{int(idx)}] must be an object",
                exit_code=EXIT_INPUT_ERROR,
            )
        normalized.append(_normalize_reference_source(entry, idx=idx, source_label=source_label))
    return normalized


def load_reference_sources_file(path: Path) -> list[dict[str, Any]]:
    """Load, validate, and normalize a ``reference_sources.v1`` file."""
    payload = load_json_file(path, label="reference sources file")
    return normalize_reference_sources_payload(payload, source_label=f"reference sources file {path}")


__all__ = [
    "REFERENCE_SOURCES_SCHEMA_VERSION",
    "load_reference_sources_file",
    "normalize_reference_sources_payload",
]
