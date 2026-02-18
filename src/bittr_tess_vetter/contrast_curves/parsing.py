"""Parsing helpers for text and FITS contrast-curve files."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from bittr_tess_vetter.api.fpp import ContrastCurve
from bittr_tess_vetter.api.fpp_helpers import load_contrast_curve_exofop_tbl
from bittr_tess_vetter.contrast_curves.exceptions import ContrastCurveParseError

_FLOAT_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")
_ALLOWED_TEXT_EXTS = {".tbl", ".dat", ".csv", ".txt"}
_ALLOWED_FITS_EXTS = {".fits", ".fit", ".fts"}


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
        for raw, norm in normalized.items():
            if norm == token_norm:
                return raw
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

