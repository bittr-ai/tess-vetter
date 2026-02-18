"""Sector-to-sector transit consistency computations (metrics-only).

This module provides lightweight computations for checking whether per-sector
transit measurements (e.g., depth) are mutually consistent.

Design notes:
- Metrics-only: no policy decisions beyond the returned classification string.
- Host apps should provide the per-sector measurements; this module does not
  fetch light curves or perform per-sector fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import stats

ConsistencyClass = Literal["EXPECTED_SCATTER", "INCONSISTENT", "UNRESOLVABLE"]


@dataclass
class SectorMeasurement:
    """Per-sector transit measurement container."""

    sector: int
    depth_ppm: float
    depth_err_ppm: float
    duration_hours: float = 0.0
    duration_err_hours: float = 0.0
    n_transits: int = 0
    shape_metric: float = 0.0
    quality_weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "sector": int(self.sector),
            "depth_ppm": float(self.depth_ppm),
            "depth_err_ppm": float(self.depth_err_ppm),
            "duration_hours": float(self.duration_hours),
            "duration_err_hours": float(self.duration_err_hours),
            "n_transits": int(self.n_transits),
            "shape_metric": float(self.shape_metric),
            "quality_weight": float(self.quality_weight),
        }


def compute_sector_consistency(
    measurements: list[SectorMeasurement],
    *,
    chi2_threshold: float = 0.01,
    min_sectors: int = 2,
) -> tuple[ConsistencyClass, list[int], float]:
    """Classify sector-to-sector consistency from precomputed measurements.

    Returns:
        (consistency_class, outlier_sectors, chi2_pval)
    """
    valid = [
        m
        for m in measurements
        if float(m.quality_weight) > 0
        and np.isfinite(float(m.depth_ppm))
        and np.isfinite(float(m.depth_err_ppm))
        and float(m.depth_err_ppm) > 0
    ]
    non_detection_outliers = [int(m.sector) for m in valid if float(m.depth_ppm) <= 0]
    positive = [m for m in valid if float(m.depth_ppm) > 0]

    if len(positive) < min_sectors:
        return "UNRESOLVABLE", sorted(set(non_detection_outliers)), 1.0

    depths = np.array([m.depth_ppm for m in positive], dtype=np.float64)
    errors = np.array([m.depth_err_ppm for m in positive], dtype=np.float64)

    if np.any(errors <= 0):
        return "UNRESOLVABLE", sorted(set(non_detection_outliers)), 1.0

    inv_var = 1.0 / (errors**2)
    weighted_mean = float(np.sum(depths * inv_var) / np.sum(inv_var))

    chi2 = float(np.sum(((depths - weighted_mean) / errors) ** 2))
    dof = len(positive) - 1
    if dof <= 0:
        return "UNRESOLVABLE", sorted(set(non_detection_outliers)), 1.0

    chi2_pval = float(stats.chi2.sf(chi2, dof))

    residuals = np.abs(depths - weighted_mean) / errors
    outlier_mask = residuals > 3.0
    residual_outliers = [int(positive[i].sector) for i in range(len(positive)) if bool(outlier_mask[i])]
    outlier_sectors = sorted(set(residual_outliers + non_detection_outliers))

    if chi2_pval < chi2_threshold:
        return "INCONSISTENT", outlier_sectors, chi2_pval
    return "EXPECTED_SCATTER", outlier_sectors, chi2_pval


__all__ = [
    "ConsistencyClass",
    "SectorMeasurement",
    "compute_sector_consistency",
]
