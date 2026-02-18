"""Shared contrast-curve parsing, normalization, and summary helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from bittr_tess_vetter.api.fpp import ContrastCurve
from bittr_tess_vetter.api.fpp_helpers import load_contrast_curve_exofop_tbl

_FLOAT_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")
_ALLOWED_TEXT_EXTS = {".tbl", ".dat", ".csv", ".txt"}
_ALLOWED_FITS_EXTS = {".fits", ".fit", ".fts"}
_MAX_RELEVANT_SEPARATION_ARCSEC = 4.0


class ContrastCurveParseError(ValueError):
    """Raised when a contrast-curve file cannot be parsed."""


def _fallback_parse_contrast_curve(path: Path, *, filter_name: str | None) -> ContrastCurve:
    text = path.read_text(encoding="utf-8", errors="replace")
    seps: list[float] = []
    dmags: list[float] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        if not (_FLOAT_RE.match(parts[0]) and _FLOAT_RE.match(parts[1])):
            continue
        try:
            sep = float(parts[0])
            dmag = float(parts[1])
        except (TypeError, ValueError):
            continue
        if np.isfinite(sep) and np.isfinite(dmag):
            seps.append(float(sep))
            dmags.append(float(dmag))

    if len(seps) < 2:
        raise ContrastCurveParseError(f"Could not parse at least two numeric rows from {path}")

    sep_arr = np.asarray(seps, dtype=np.float64)
    dmag_arr = np.asarray(dmags, dtype=np.float64)
    order = np.argsort(sep_arr)
    return ContrastCurve(
        separation_arcsec=sep_arr[order],
        delta_mag=dmag_arr[order],
        filter=str(filter_name) if filter_name is not None else "Vis",
    )


def _normalize_colname(name: str) -> str:
    text = str(name).strip().lower()
    for ch in (" ", "-", "/", "(", ")", "[", "]", "{", "}", ".", ":"):
        text = text.replace(ch, "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def _find_column_name(names: list[str], *, tokens: tuple[str, ...]) -> str | None:
    normalized = {name: _normalize_colname(name) for name in names}
    for token in tokens:
        token_norm = _normalize_colname(token)
        # Exact first.
        for raw, norm in normalized.items():
            if norm == token_norm:
                return raw
        # Substring fallback.
        for raw, norm in normalized.items():
            if token_norm in norm:
                return raw
    return None


def _parse_fits_contrast_curve(path: Path, *, filter_name: str | None) -> ContrastCurve:
    try:
        from astropy.io import fits
    except Exception as exc:
        raise ContrastCurveParseError(
            "FITS contrast-curve parsing requires astropy.io.fits to be available."
        ) from exc

    sep_tokens = (
        "separation_arcsec",
        "separation",
        "sep_arcsec",
        "sep",
        "rho",
        "distance",
        "arcsec",
        "arc_sec",
    )
    dmag_tokens = (
        "delta_mag",
        "deltamag",
        "d_mag",
        "dmag",
        "mag_diff",
        "magdiff",
        "contrast",
    )

    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            table = getattr(hdu, "data", None)
            names = list(getattr(table, "names", []) or [])
            if table is None or not names:
                continue

            sep_col = _find_column_name(names, tokens=sep_tokens)
            dmag_col = _find_column_name(names, tokens=dmag_tokens)
            if sep_col is None or dmag_col is None:
                continue

            try:
                sep_arr = np.asarray(table[sep_col], dtype=np.float64)
                dmag_arr = np.asarray(table[dmag_col], dtype=np.float64)
            except Exception:
                continue

            finite = np.isfinite(sep_arr) & np.isfinite(dmag_arr)
            sep_arr = sep_arr[finite]
            dmag_arr = dmag_arr[finite]
            if sep_arr.size < 2:
                continue

            # Heuristic unit normalization: many high-res products store separation in mas.
            sep_max = float(np.nanmax(sep_arr))
            if sep_max > 60.0 and sep_max <= 60_000.0:
                sep_arr = sep_arr / 1000.0

            order = np.argsort(sep_arr)
            return ContrastCurve(
                separation_arcsec=sep_arr[order],
                delta_mag=dmag_arr[order],
                filter=str(filter_name) if filter_name is not None else "Vis",
            )

    raise ContrastCurveParseError(
        f"FITS file does not contain a parseable contrast-curve table with separation and delta-mag columns: {path}"
    )


def parse_contrast_curve_file(path: str | Path, *, filter_name: str | None = None) -> ContrastCurve:
    """Parse local contrast-curve file using API helper with a generic fallback."""
    parsed_path = Path(path).expanduser().resolve()
    if not parsed_path.exists() or not parsed_path.is_file():
        raise FileNotFoundError(f"Contrast curve file not found: {parsed_path}")
    suffix = parsed_path.suffix.lower()
    if suffix in _ALLOWED_FITS_EXTS:
        return _parse_fits_contrast_curve(parsed_path, filter_name=filter_name)
    if suffix not in _ALLOWED_TEXT_EXTS:
        raise ContrastCurveParseError(
            f"Unsupported contrast-curve extension for parser: {parsed_path.suffix} (file={parsed_path})"
        )

    try:
        return load_contrast_curve_exofop_tbl(parsed_path, filter=filter_name)
    except Exception as primary_error:
        try:
            return _fallback_parse_contrast_curve(parsed_path, filter_name=filter_name)
        except Exception as fallback_error:
            raise ContrastCurveParseError(
                f"Failed to parse contrast curve file {parsed_path}: {primary_error}; fallback failed: {fallback_error}"
            ) from fallback_error


def normalize_contrast_curve(curve: ContrastCurve) -> dict[str, Any]:
    """Normalize arrays and compute reusable scalar summaries."""
    sep_arr = np.asarray(curve.separation_arcsec, dtype=np.float64)
    dmag_arr = np.asarray(curve.delta_mag, dtype=np.float64)

    finite = np.isfinite(sep_arr) & np.isfinite(dmag_arr)
    sep_arr = sep_arr[finite]
    dmag_arr = dmag_arr[finite]

    if sep_arr.size < 2:
        raise ContrastCurveParseError("Contrast curve has fewer than two finite points")

    keep = sep_arr >= 0.0
    sep_arr = sep_arr[keep]
    dmag_arr = dmag_arr[keep]
    if sep_arr.size < 2:
        raise ContrastCurveParseError("Contrast curve has fewer than two non-negative separation points")

    order = np.argsort(sep_arr)
    sep_arr = sep_arr[order]
    dmag_arr = dmag_arr[order]
    unique_sep = np.unique(sep_arr)
    if unique_sep.size < sep_arr.size:
        merged_dmag = np.zeros(unique_sep.shape, dtype=np.float64)
        for idx, sep in enumerate(unique_sep):
            mask = sep_arr == sep
            merged_dmag[idx] = float(np.nanmax(dmag_arr[mask]))
        sep_arr = unique_sep
        dmag_arr = merged_dmag
    original_sep_max = float(np.max(sep_arr))
    original_n_points = int(sep_arr.size)

    # Guardrails against corrupted or unit-mismatched parses.
    sep_max = float(np.max(sep_arr))
    if sep_max > 60.0:
        raise ContrastCurveParseError(
            f"Implausible separation scale detected (max={sep_max:.3f} arcsec)."
        )

    # Truncate to astrophysically relevant inner working angles before interpolation/ruling.
    # Outer-tail contrast extrapolations in AO products can become non-physical and are
    # not informative for TRICERATOPS priors in this workflow.
    inner_mask = sep_arr <= float(_MAX_RELEVANT_SEPARATION_ARCSEC)
    if int(np.sum(inner_mask)) < 2:
        raise ContrastCurveParseError(
            f"Fewer than 2 data points within {_MAX_RELEVANT_SEPARATION_ARCSEC:.1f} arcsec."
        )
    sep_arr = sep_arr[inner_mask]
    dmag_arr = dmag_arr[inner_mask]
    truncated_n_points = original_n_points - int(sep_arr.size)
    truncation_applied = truncated_n_points > 0
    dmag_max = float(np.max(dmag_arr))
    dmag_min = float(np.min(dmag_arr))
    if dmag_max > 25.0 or dmag_min < -2.0:
        raise ContrastCurveParseError(
            f"Implausible delta_mag range detected (min={dmag_min:.3f}, max={dmag_max:.3f})."
        )

    flux_ratio = np.power(10.0, -0.4 * dmag_arr)

    return {
        "separation_arcsec": [float(x) for x in sep_arr],
        "delta_mag": [float(x) for x in dmag_arr],
        "delta_flux_ratio": [float(x) for x in flux_ratio],
        "n_points": int(sep_arr.size),
        "separation_arcsec_min": float(np.min(sep_arr)),
        "separation_arcsec_max": float(np.max(sep_arr)),
        "delta_mag_min": float(np.min(dmag_arr)),
        "delta_mag_max": float(np.max(dmag_arr)),
        "truncation_applied": bool(truncation_applied),
        "truncation_max_separation_arcsec": float(_MAX_RELEVANT_SEPARATION_ARCSEC),
        "n_points_before_truncation": int(original_n_points),
        "n_points_dropped_by_truncation": int(truncated_n_points),
        "original_separation_arcsec_max": float(original_sep_max),
    }


def build_ruling_summary(normalized: dict[str, Any]) -> dict[str, Any]:
    """Build compact rule-based coverage summary from normalized arrays."""
    sep = np.asarray(normalized.get("separation_arcsec", []), dtype=np.float64)
    dmag = np.asarray(normalized.get("delta_mag", []), dtype=np.float64)
    if sep.size < 2 or dmag.size < 2:
        return {
            "status": "no_data",
            "quality_assessment": "none",
            "notes": ["No usable contrast-curve points available."],
        }

    dmag_at_05 = float(np.interp(0.5, sep, dmag))
    dmag_at_10 = float(np.interp(1.0, sep, dmag))
    dmag_at_20 = float(np.interp(2.0, sep, dmag))

    quality_assessment = "shallow"
    if dmag_at_05 >= 6.5:
        quality_assessment = "deep"
    elif dmag_at_05 >= 4.0:
        quality_assessment = "moderate"

    notes = [
        "Higher delta_mag means stronger exclusion of faint companions.",
        "Use strongest available curve when running FPP with contrast constraints.",
    ]
    if bool(normalized.get("truncation_applied")):
        notes.append(
            f"Curve was truncated to <= {float(normalized.get('truncation_max_separation_arcsec', _MAX_RELEVANT_SEPARATION_ARCSEC)):.1f} arcsec for plausibility."
        )
    extrapolated = []
    sep_min = float(np.min(sep))
    sep_max = float(np.max(sep))
    for probe in (0.5, 1.0, 2.0):
        if probe < sep_min or probe > sep_max:
            extrapolated.append(probe)
    if extrapolated:
        notes.append(
            "One or more summary depths are extrapolated outside sampled separations: "
            + ", ".join(f"{x:.1f}\"" for x in extrapolated)
            + "."
        )

    return {
        "status": "ok",
        "quality_assessment": quality_assessment,
        "max_delta_mag_at_0p5_arcsec": dmag_at_05,
        "max_delta_mag_at_1p0_arcsec": dmag_at_10,
        "max_delta_mag_at_2p0_arcsec": dmag_at_20,
        "innermost_separation_arcsec": sep_min,
        "outermost_separation_arcsec": sep_max,
        # Backward-compatible aliases used by existing intermediate consumers.
        "dmag_at_0p5_arcsec": dmag_at_05,
        "dmag_at_1p0_arcsec": dmag_at_10,
        "dmag_at_2p0_arcsec": dmag_at_20,
        "inner_working_angle_arcsec": sep_min,
        "outer_working_angle_arcsec": sep_max,
        "notes": notes,
    }


def derive_contrast_verdict(ruling_summary: dict[str, Any]) -> tuple[str, str]:
    """Return canonical verdict token and JSON path source for contrast outputs."""
    status = str(ruling_summary.get("status") or "").lower()
    if status != "ok":
        return "NO_CONTRAST_CURVES", "$.ruling_summary.status"

    quality = str(ruling_summary.get("quality_assessment") or "").lower()
    if quality == "deep":
        return "CONTRAST_CURVES_STRONG", "$.ruling_summary.quality_assessment"
    if quality == "moderate":
        return "CONTRAST_CURVES_MODERATE", "$.ruling_summary.quality_assessment"
    return "CONTRAST_CURVES_LIMITED", "$.ruling_summary.quality_assessment"


def combine_normalized_curves(curves: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Combine multiple normalized curves into a single exclusion envelope."""
    if not curves:
        return None

    all_sep: list[float] = []
    for curve in curves:
        all_sep.extend(float(x) for x in curve.get("separation_arcsec", []))
    if len(all_sep) < 2:
        return None

    grid = np.unique(np.asarray(all_sep, dtype=np.float64))
    if grid.size < 2:
        return None

    interpolated_rows: list[np.ndarray] = []
    support_counts = np.zeros(grid.shape, dtype=np.int32)
    for curve in curves:
        sep = np.asarray(curve.get("separation_arcsec", []), dtype=np.float64)
        dmag = np.asarray(curve.get("delta_mag", []), dtype=np.float64)
        if sep.size < 2 or dmag.size < 2:
            continue
        interp = np.interp(grid, sep, dmag)
        support_mask = (grid >= np.min(sep)) & (grid <= np.max(sep))
        interp = np.where(support_mask, interp, np.nan)
        support_counts = support_counts + support_mask.astype(np.int32)
        interpolated_rows.append(interp)

    if not interpolated_rows:
        return None

    stack = np.vstack(interpolated_rows)
    with np.errstate(invalid="ignore"):
        combined_dmag = np.nanmax(stack, axis=0)
    keep = np.isfinite(combined_dmag)
    if int(np.sum(keep)) < 2:
        return None

    grid_kept = grid[keep]
    dmag_kept = combined_dmag[keep]
    flux_kept = np.power(10.0, -0.4 * dmag_kept)

    return {
        "separation_arcsec": [float(x) for x in grid_kept],
        "delta_mag": [float(x) for x in dmag_kept],
        "delta_flux_ratio": [float(x) for x in flux_kept],
        "supporting_observations": [int(x) for x in support_counts[keep]],
        "n_points": int(grid_kept.size),
        "n_observations": int(len(interpolated_rows)),
    }


__all__ = [
    "ContrastCurveParseError",
    "build_ruling_summary",
    "combine_normalized_curves",
    "derive_contrast_verdict",
    "normalize_contrast_curve",
    "parse_contrast_curve_file",
]
