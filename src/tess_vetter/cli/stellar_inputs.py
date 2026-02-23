"""Shared CLI helpers for resolving stellar inputs with explicit precedence."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from tess_vetter.cli.common_cli import (
    EXIT_DATA_UNAVAILABLE,
    EXIT_INPUT_ERROR,
    BtvCliError,
    load_json_file,
)
from tess_vetter.platform.catalogs.exofop_toi_table import fetch_exofop_toi_table
from tess_vetter.platform.io.mast_client import MASTClient

_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "radius": ("radius", "stellar_radius", "stellar_radius_rsun"),
    "mass": ("mass", "stellar_mass", "stellar_mass_msun"),
    "tmag": ("tmag", "stellar_tmag"),
}

_EXOFOP_STELLAR_ALIASES: dict[str, tuple[str, ...]] = {
    "radius": ("stellar_radius_r_sun", "stellar_radius", "radius_r_sun", "radius"),
    "mass": ("stellar_mass_m_sun", "stellar_mass", "mass_m_sun", "mass"),
    "tmag": ("tess_mag", "tmag", "stellar_tmag"),
    "teff": ("stellar_eff_temp_k", "teff", "teff_k"),
    "logg": ("stellar_log_g_cm_s^2", "stellar_log_g", "logg", "logg_cgs"),
}


def _safe_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        out = float(text)
    except Exception:
        return None
    return out if np.isfinite(out) else None


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
    auto_meta: dict[str, Any] | None = None
    if use_stellar_auto:
        if auto_loader is None:
            raise BtvCliError("stellar auto lookup is unavailable", exit_code=EXIT_DATA_UNAVAILABLE)
        auto_raw = auto_loader(int(tic_id))
        if auto_raw is not None:
            if isinstance(auto_raw, tuple):
                auto_payload, raw_meta = auto_raw
                auto_values = _normalize_stellar_mapping(auto_payload, label="auto stellar")
                auto_meta = dict(raw_meta) if isinstance(raw_meta, dict) else None
            else:
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
        "auto": auto_meta,
    }
    return resolved, provenance


def _lookup_exofop_stellar_values(tic_id: int, *, toi: str | None = None) -> tuple[dict[str, float | None], dict[str, Any]]:
    rows = fetch_exofop_toi_table().entries_for_tic(int(tic_id))
    if toi is not None:
        toi_norm = str(toi).upper().replace("TOI-", "").replace("TOI", "").strip()
        filtered = [r for r in rows if str(r.get("toi", "")).strip() == toi_norm]
        if filtered:
            rows = filtered
    if not rows:
        return {"radius": None, "mass": None, "tmag": None}, {
            "source": "exofop_toi_table",
            "status": "data_unavailable",
            "message": f"No ExoFOP TOI rows for TIC {int(tic_id)}",
        }

    row = dict(rows[0])
    out: dict[str, float | None] = {"radius": None, "mass": None, "tmag": None}
    for field, aliases in _EXOFOP_STELLAR_ALIASES.items():
        for alias in aliases:
            value = row.get(alias)
            if value is None:
                continue
            try:
                num = float(value)
            except Exception:
                continue
            if np.isfinite(num):
                if field in out:
                    out[field] = num
                break

    return out, {
        "source": "exofop_toi_table",
        "status": "ok",
        "toi": row.get("toi"),
        "teff": _safe_optional_float(row.get("stellar_eff_temp_k")),
        "logg": _safe_optional_float(row.get("stellar_log_g_cm_s^2")),
    }


def load_auto_stellar_with_fallback(
    *,
    tic_id: int,
    toi: str | None = None,
) -> tuple[dict[str, float | None], dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    tic_values: dict[str, float | None] = {"radius": None, "mass": None, "tmag": None}
    tic_ok = False
    try:
        target = MASTClient().get_target_info(int(tic_id))
        tic_values = {
            "radius": _coerce_optional_finite_float(target.stellar.radius, label="tic radius"),
            "mass": _coerce_optional_finite_float(target.stellar.mass, label="tic mass"),
            "tmag": _coerce_optional_finite_float(target.stellar.tmag, label="tic tmag"),
        }
        tic_ok = True
        attempts.append({"source": "tic_mast", "status": "ok"})
    except Exception as exc:
        attempts.append({"source": "tic_mast", "status": "error", "message": f"{type(exc).__name__}: {exc}"})

    exo_values, exo_meta = _lookup_exofop_stellar_values(int(tic_id), toi=toi)
    attempts.append({k: v for k, v in exo_meta.items() if k in {"source", "status", "message", "toi"}})

    resolved = dict(tic_values)
    field_sources: dict[str, str] = {}
    for field in ("radius", "mass", "tmag"):
        if resolved.get(field) is None and exo_values.get(field) is not None:
            resolved[field] = exo_values[field]
            field_sources[field] = "exofop_toi_table"
        elif resolved.get(field) is not None:
            field_sources[field] = "tic_mast"
        else:
            field_sources[field] = "missing"

    same_as_tic = False
    if tic_ok:
        overlap = [
            field
            for field in ("radius", "mass", "tmag")
            if tic_values.get(field) is not None and exo_values.get(field) is not None
        ]
        same_as_tic = bool(overlap) and all(
            abs(float(tic_values[field]) - float(exo_values[field])) <= 1e-9 for field in overlap
        )

    return resolved, {
        "selected_source": "tic_mast" if tic_ok else "exofop_toi_table",
        "field_sources": field_sources,
        "attempts": attempts,
        "echo_of_tic": same_as_tic,
        "exofop": exo_meta,
    }


__all__ = [
    "load_stellar_inputs_file",
    "resolve_stellar_inputs",
    "load_auto_stellar_with_fallback",
]
