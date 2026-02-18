"""Parsing helpers for text and FITS contrast-curve files."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from bittr_tess_vetter.api.fpp import ContrastCurve
from bittr_tess_vetter.api.fpp_helpers import load_contrast_curve_exofop_tbl
from bittr_tess_vetter.contrast_curves.exceptions import ContrastCurveParseError

_FLOAT_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")
_ALLOWED_TEXT_EXTS = {".tbl", ".dat", ".csv", ".txt"}
_ALLOWED_FITS_EXTS = {".fits", ".fit", ".fts"}
_FITS_IMAGE_LOOKUP_PIXEL_SCALE_ARCSEC = (
    (re.compile(r"(?i).*_soarspeckle\.fits$"), 0.01575, "lookup:soar_hrcam"),
    (re.compile(r"(?i).*_sharcs_ao_.*\.fits$"), 0.033, "lookup:sharcs"),
)
_FITS_FILTER_NAME_PATTERNS = (
    (re.compile(r"(?i).*[._-]j(?:\.fits|$)"), "J"),
    (re.compile(r"(?i).*[._-]ks?(?:\.fits|$)"), "Ks"),
    (re.compile(r"(?i).*[._-]kcont(?:\.fits|$)"), "Kcont"),
    (re.compile(r"(?i).*[._-]562(?:\.fits|$)"), "562nm"),
    (re.compile(r"(?i).*[._-]832(?:\.fits|$)"), "832nm"),
    (re.compile(r"(?i).*[._-]b(?:\.fits|$)"), "562nm"),
    (re.compile(r"(?i).*[._-]r(?:\.fits|$)"), "832nm"),
)
_DEFAULT_SIGMA_THRESHOLD = 5.0
_DEFAULT_ANNULUS_WIDTH_PX = 3
_DEFAULT_MIN_PIXELS_PER_ANNULUS = 10
_DEFAULT_INNER_RADIUS_PX = 3
_DEFAULT_CORE_RADIUS_PX = 2
_DEFAULT_MAX_SEPARATION_ARCSEC = 4.0


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


def _parse_fits_contrast_curve(
    path: Path, *, filter_name: str | None, pixel_scale_arcsec_per_px: float | None = None
) -> tuple[ContrastCurve, dict[str, Any]]:
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
            return (
                ContrastCurve(
                    separation_arcsec=sep_arr[order],
                    delta_mag=dmag_arr[order],
                    filter=str(filter_name) if filter_name is not None else "Vis",
                ),
                {"strategy": "fits_table"},
            )

        for hdu in hdul:
            image = np.asarray(getattr(hdu, "data", None))
            if image.ndim != 2:
                continue
            if image.shape[0] < 16 or image.shape[1] < 16:
                continue
            provenance: dict[str, Any] = {}
            pixel_scale, pixel_scale_source = _resolve_pixel_scale_arcsec_per_px(
                header=getattr(hdu, "header", {}),
                path=path,
                pixel_scale_override_arcsec_per_px=pixel_scale_arcsec_per_px,
            )
            if pixel_scale is None:
                continue
            sep_arr, dmag_arr = _extract_azimuthal_contrast_curve(
                image=image,
                pixel_scale_arcsec_per_px=float(pixel_scale),
                provenance=provenance,
            )
            if sep_arr.size < 2:
                continue
            order = np.argsort(sep_arr)
            resolved_filter = _resolve_filter_label(
                explicit_filter=filter_name,
                header=getattr(hdu, "header", {}),
                path=path,
            )
            curve = ContrastCurve(
                separation_arcsec=sep_arr[order],
                delta_mag=dmag_arr[order],
                filter=resolved_filter,
            )
            return (
                curve,
                {
                    "strategy": "fits_image_azimuthal",
                    "pixel_scale_arcsec_per_px": float(pixel_scale),
                    "pixel_scale_source": pixel_scale_source,
                    **provenance,
                },
            )

    raise ContrastCurveParseError(
        f"FITS file does not contain a parseable contrast-curve table or valid 2D image for azimuthal extraction: {path}"
    )


def _resolve_filter_label(*, explicit_filter: str | None, header: Any, path: Path) -> str:
    if explicit_filter is not None:
        return str(explicit_filter)
    header_filter = str(getattr(header, "get", lambda *_: None)("FILTER") or "").strip()
    if header_filter:
        return header_filter
    filename = path.name
    for pattern, label in _FITS_FILTER_NAME_PATTERNS:
        if pattern.match(filename):
            return label
    return "Vis"


def _resolve_pixel_scale_arcsec_per_px(
    *,
    header: Any,
    path: Path,
    pixel_scale_override_arcsec_per_px: float | None = None,
) -> tuple[float | None, str | None]:
    if pixel_scale_override_arcsec_per_px is not None:
        try:
            parsed = float(pixel_scale_override_arcsec_per_px)
        except (TypeError, ValueError):
            parsed = float("nan")
        if np.isfinite(parsed) and parsed > 0.0:
            return float(parsed), "override:cli"
    header_get = getattr(header, "get", None)
    if callable(header_get):
        for key in ("PIXSCL", "PIXSCAL", "PIXELSCAL", "PIXSCALE"):
            value = header_get(key)
            if value is None:
                continue
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(parsed) and parsed > 0.0:
                return float(parsed), f"header:{key}"
        cdelt1 = header_get("CDELT1")
        cdelt2 = header_get("CDELT2")
        for key, value in (("CDELT1", cdelt1), ("CDELT2", cdelt2)):
            try:
                parsed = abs(float(value)) * 3600.0
            except (TypeError, ValueError):
                continue
            if np.isfinite(parsed) and parsed > 0.0:
                return float(parsed), f"header:{key}"
        # WCS CD matrix can encode scale without PIXSCL/CDELT keywords.
        cd11 = header_get("CD1_1")
        cd12 = header_get("CD1_2")
        cd21 = header_get("CD2_1")
        cd22 = header_get("CD2_2")
        try:
            row1 = np.sqrt(float(cd11) ** 2 + float(cd12) ** 2)
            row2 = np.sqrt(float(cd21) ** 2 + float(cd22) ** 2)
            parsed = 0.5 * (abs(row1) + abs(row2)) * 3600.0
        except (TypeError, ValueError):
            parsed = float("nan")
        if np.isfinite(parsed) and parsed > 0.0:
            return float(parsed), "header:CD_matrix"

    filename = path.name
    for pattern, value, source in _FITS_IMAGE_LOOKUP_PIXEL_SCALE_ARCSEC:
        if pattern.match(filename):
            return float(value), source
    return None, None


def _extract_azimuthal_contrast_curve(
    *,
    image: np.ndarray,
    pixel_scale_arcsec_per_px: float,
    provenance: dict[str, Any] | None = None,
    sigma_threshold: float = _DEFAULT_SIGMA_THRESHOLD,
    annulus_width_px: int = _DEFAULT_ANNULUS_WIDTH_PX,
    min_pixels_per_annulus: int = _DEFAULT_MIN_PIXELS_PER_ANNULUS,
    inner_radius_px: int = _DEFAULT_INNER_RADIUS_PX,
    core_radius_px: int = _DEFAULT_CORE_RADIUS_PX,
    max_separation_arcsec: float = _DEFAULT_MAX_SEPARATION_ARCSEC,
) -> tuple[np.ndarray, np.ndarray]:
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ContrastCurveParseError("FITS image extraction requires a 2D array.")
    finite_mask = np.isfinite(img)
    if int(np.sum(finite_mask)) < 50:
        raise ContrastCurveParseError("FITS image extraction failed: insufficient finite pixels.")

    masked = np.where(finite_mask, img, -np.inf)
    peak_index = int(np.argmax(masked))
    y0, x0 = np.unravel_index(peak_index, img.shape)

    y_min = max(0, y0 - int(core_radius_px))
    y_max = min(img.shape[0], y0 + int(core_radius_px) + 1)
    x_min = max(0, x0 - int(core_radius_px))
    x_max = min(img.shape[1], x0 + int(core_radius_px) + 1)
    core = img[y_min:y_max, x_min:x_max]
    core_finite = np.isfinite(core)
    if int(np.sum(core_finite)) < 3:
        raise ContrastCurveParseError("FITS image extraction failed: core aperture has insufficient finite pixels.")
    f_star = float(np.nansum(core[core_finite]))
    if not np.isfinite(f_star) or f_star <= 0.0:
        raise ContrastCurveParseError("FITS image extraction failed: invalid reference stellar flux.")

    yy, xx = np.indices(img.shape)
    rr = np.sqrt((xx - float(x0)) ** 2 + (yy - float(y0)) ** 2)
    half_width = 0.5 * float(min(img.shape))
    r_max_px = min(float(max_separation_arcsec) / float(pixel_scale_arcsec_per_px), 0.8 * half_width)
    if not np.isfinite(r_max_px) or r_max_px <= float(inner_radius_px + annulus_width_px):
        raise ContrastCurveParseError("FITS image extraction failed: insufficient radial extent.")

    sep_values: list[float] = []
    dmag_values: list[float] = []
    annulus_attempts = 0
    annulus_used = 0
    annulus_skipped_lowpix = 0
    annulus_skipped_sigma = 0
    r_inner = float(inner_radius_px)
    while r_inner + float(annulus_width_px) <= r_max_px:
        annulus_attempts += 1
        r_outer = r_inner + float(annulus_width_px)
        annulus_mask = (rr >= r_inner) & (rr < r_outer) & finite_mask
        pixel_count = int(np.sum(annulus_mask))
        if pixel_count < int(min_pixels_per_annulus):
            annulus_skipped_lowpix += 1
            r_inner = r_outer
            continue
        values = img[annulus_mask]
        sigma = float(np.nanstd(values, ddof=1))
        if not np.isfinite(sigma) or sigma <= 0.0:
            annulus_skipped_sigma += 1
            r_inner = r_outer
            continue
        ratio = float(sigma_threshold) * sigma / f_star
        if not np.isfinite(ratio) or ratio <= 0.0:
            annulus_skipped_sigma += 1
            r_inner = r_outer
            continue
        dmag = float(-2.5 * np.log10(ratio))
        separation_arcsec = float((r_inner + r_outer) * 0.5 * pixel_scale_arcsec_per_px)
        if np.isfinite(dmag) and np.isfinite(separation_arcsec):
            dmag_values.append(dmag)
            sep_values.append(separation_arcsec)
            annulus_used += 1
        r_inner = r_outer

    sep_arr = np.asarray(sep_values, dtype=np.float64)
    dmag_arr = np.asarray(dmag_values, dtype=np.float64)
    if sep_arr.size < 2:
        raise ContrastCurveParseError(
            "FITS image extraction failed: fewer than two valid annuli for contrast estimation."
        )
    if provenance is not None:
        provenance.update(
            {
                "center_pixel": [int(y0), int(x0)],
                "f_star_core_sum": float(f_star),
                "sigma_threshold": float(sigma_threshold),
                "annulus_width_px": int(annulus_width_px),
                "min_pixels_per_annulus": int(min_pixels_per_annulus),
                "inner_radius_px": int(inner_radius_px),
                "core_radius_px": int(core_radius_px),
                "max_separation_arcsec": float(max_separation_arcsec),
                "annulus_attempts": int(annulus_attempts),
                "annulus_used": int(annulus_used),
                "annulus_skipped_low_pixels": int(annulus_skipped_lowpix),
                "annulus_skipped_sigma": int(annulus_skipped_sigma),
            }
        )
    return sep_arr, dmag_arr


def parse_contrast_curve_with_provenance(
    path: str | Path,
    *,
    filter_name: str | None = None,
    pixel_scale_arcsec_per_px: float | None = None,
) -> tuple[ContrastCurve, dict[str, Any]]:
    """Parse local contrast-curve file and return curve + parse provenance."""
    parsed_path = Path(path).expanduser().resolve()
    if not parsed_path.exists() or not parsed_path.is_file():
        raise FileNotFoundError(f"Contrast curve file not found: {parsed_path}")
    suffix = parsed_path.suffix.lower()
    if suffix in _ALLOWED_FITS_EXTS:
        curve, provenance = _parse_fits_contrast_curve(
            parsed_path,
            filter_name=filter_name,
            pixel_scale_arcsec_per_px=pixel_scale_arcsec_per_px,
        )
        return curve, provenance
    if suffix not in _ALLOWED_TEXT_EXTS:
        raise ContrastCurveParseError(
            f"Unsupported contrast-curve extension for parser: {parsed_path.suffix} (file={parsed_path})"
        )

    try:
        curve = load_contrast_curve_exofop_tbl(parsed_path, filter=filter_name)
        return curve, {"strategy": "text_table_primary"}
    except Exception as primary_error:
        try:
            curve = _fallback_parse_contrast_curve(parsed_path, filter_name=filter_name)
            return curve, {"strategy": "text_table_fallback"}
        except Exception as fallback_error:
            raise ContrastCurveParseError(
                f"Failed to parse contrast curve file {parsed_path}: {primary_error}; fallback failed: {fallback_error}"
            ) from fallback_error


def parse_contrast_curve_file(
    path: str | Path,
    *,
    filter_name: str | None = None,
    pixel_scale_arcsec_per_px: float | None = None,
) -> ContrastCurve:
    """Parse local contrast-curve file using API helper with a generic fallback."""
    curve, _ = parse_contrast_curve_with_provenance(
        path,
        filter_name=filter_name,
        pixel_scale_arcsec_per_px=pixel_scale_arcsec_per_px,
    )
    return curve
