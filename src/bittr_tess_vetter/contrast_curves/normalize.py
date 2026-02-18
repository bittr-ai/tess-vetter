"""Normalization and plausibility checks for contrast curves."""

from __future__ import annotations

import numpy as np

from bittr_tess_vetter.api.fpp import ContrastCurve
from bittr_tess_vetter.contrast_curves.exceptions import ContrastCurveParseError
from bittr_tess_vetter.contrast_curves.models import NormalizedContrastCurve

MAX_RELEVANT_SEPARATION_ARCSEC = 4.0


def normalize_contrast_curve(curve: ContrastCurve) -> NormalizedContrastCurve:
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

    sep_max = float(np.max(sep_arr))
    if sep_max > 60.0:
        raise ContrastCurveParseError(
            f"Implausible separation scale detected (max={sep_max:.3f} arcsec)."
        )

    inner_mask = sep_arr <= float(MAX_RELEVANT_SEPARATION_ARCSEC)
    if int(np.sum(inner_mask)) < 2:
        raise ContrastCurveParseError(
            f"Fewer than 2 data points within {MAX_RELEVANT_SEPARATION_ARCSEC:.1f} arcsec."
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
        "truncation_max_separation_arcsec": float(MAX_RELEVANT_SEPARATION_ARCSEC),
        "n_points_before_truncation": int(original_n_points),
        "n_points_dropped_by_truncation": int(truncated_n_points),
        "original_separation_arcsec_max": float(original_sep_max),
    }

