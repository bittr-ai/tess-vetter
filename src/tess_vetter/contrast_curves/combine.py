"""Helpers to combine multiple normalized contrast curves."""

from __future__ import annotations

from typing import Any

import numpy as np


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

