"""Shared CLI helpers for resolving stellar inputs with explicit precedence."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from bittr_tess_vetter.cli.common_cli import EXIT_DATA_UNAVAILABLE, EXIT_INPUT_ERROR, BtvCliError, load_json_file

_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "radius": ("radius", "stellar_radius", "stellar_radius_rsun"),
    "mass": ("mass", "stellar_mass", "stellar_mass_msun"),
    "tmag": ("tmag", "stellar_tmag"),
}


def _coerce_optional_finite_float(value: Any, *, label: str) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise BtvCliError(f"{label} must be numeric", exit_code=EXIT_INPUT_ERROR) from exc
    if not np.isfinite(out):
        raise BtvCliError(f"{label} must be finite", exit_code=EXIT_INPUT_ERROR)
    return out


def _normalize_stellar_mapping(raw: Mapping[str, Any], *, label: str) -> dict[str, float | None]:
    resolved: dict[str, float | None] = {"radius": None, "mass": None, "tmag": None}
    for field, aliases in _FIELD_ALIASES.items():
        chosen: Any = None
        for alias in aliases:
            if alias in raw:
                chosen = raw.get(alias)
                break
        resolved[field] = _coerce_optional_finite_float(chosen, label=f"{label} {field}")

    for positive in ("radius", "mass"):
        value = resolved.get(positive)
        if value is not None and value <= 0.0:
            raise BtvCliError(f"{label} {positive} must be > 0", exit_code=EXIT_INPUT_ERROR)
    return resolved


def load_stellar_inputs_file(path: str | Path) -> tuple[dict[str, float | None], dict[str, Any]]:
    """Load a stellar input JSON file and normalize supported fields."""
    path_obj = Path(path)
    payload = load_json_file(path_obj, label="stellar file")

    source = payload
    for nested_key in ("stellar", "stellar_params"):
        nested = payload.get(nested_key)
        if isinstance(nested, dict):
            source = nested
            break
    if not isinstance(source, dict):
        raise BtvCliError("stellar file must contain an object payload", exit_code=EXIT_INPUT_ERROR)

    normalized = _normalize_stellar_mapping(source, label=f"stellar file {path_obj}")
    return normalized, {"path": str(path_obj)}


def resolve_stellar_inputs(
    *,
    tic_id: int,
    stellar_radius: float | None,
    stellar_mass: float | None,
    stellar_tmag: float | None,
    stellar_file: str | None,
    use_stellar_auto: bool,
    require_stellar: bool,
    auto_loader: Callable[[int], Mapping[str, Any] | None] | None = None,
) -> tuple[dict[str, float | None], dict[str, Any]]:
    """Resolve stellar fields with precedence explicit > file > auto."""
    explicit = _normalize_stellar_mapping(
        {"radius": stellar_radius, "mass": stellar_mass, "tmag": stellar_tmag},
        label="explicit stellar",
    )

    file_values: dict[str, float | None] = {"radius": None, "mass": None, "tmag": None}
    file_meta: dict[str, Any] | None = None
    if stellar_file:
        file_values, file_meta = load_stellar_inputs_file(stellar_file)

    auto_values: dict[str, float | None] = {"radius": None, "mass": None, "tmag": None}
    if use_stellar_auto:
        if auto_loader is None:
            raise BtvCliError("stellar auto lookup is unavailable", exit_code=EXIT_DATA_UNAVAILABLE)
        auto_raw = auto_loader(int(tic_id))
        if auto_raw is not None:
            auto_values = _normalize_stellar_mapping(auto_raw, label="auto stellar")

    resolved: dict[str, float | None] = {"radius": None, "mass": None, "tmag": None}
    sources: dict[str, str] = {"radius": "missing", "mass": "missing", "tmag": "missing"}
    for field in ("radius", "mass", "tmag"):
        explicit_value = explicit.get(field)
        file_value = file_values.get(field)
        auto_value = auto_values.get(field)
        if explicit_value is not None:
            resolved[field] = float(explicit_value)
            sources[field] = "explicit"
        elif file_value is not None:
            resolved[field] = float(file_value)
            sources[field] = "file"
        elif auto_value is not None:
            resolved[field] = float(auto_value)
            sources[field] = "auto"

    if require_stellar and (resolved["radius"] is None or resolved["mass"] is None):
        raise BtvCliError(
            "Stellar inputs required; provide radius and mass via --stellar-* / --stellar-file / --use-stellar-auto.",
            exit_code=EXIT_DATA_UNAVAILABLE,
        )

    provenance: dict[str, Any] = {
        "values": resolved,
        "sources": sources,
        "file": file_meta,
        "use_stellar_auto": bool(use_stellar_auto),
    }
    return resolved, provenance


__all__ = ["load_stellar_inputs_file", "resolve_stellar_inputs"]
