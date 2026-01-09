"""Transit model API surface (host-facing).

This module provides stable, deterministic transit model computations for host
applications. It intentionally returns metrics only (not model arrays) to keep
responses lightweight for tool layers.
"""

from __future__ import annotations

import numpy as np

from bittr_tess_vetter.api.references import KOVACS_2002, cite, cites
from bittr_tess_vetter.compute.periodogram import compute_bls_model


@cites(cite(KOVACS_2002, "Box-shaped transit model (BLS lineage)"))
def compute_transit_model(
    *,
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    period: float,
    t0: float,
    duration_hours: float,
    depth_ppm: float,
) -> dict[str, float | int]:
    """Compute a simple box transit model and fit diagnostics.

    Returns metrics only (no model array) for host/MCP use.
    """
    time_arr = np.asarray(time, dtype=np.float64)
    flux_arr = np.asarray(flux, dtype=np.float64)
    flux_err_arr = np.asarray(flux_err, dtype=np.float64)

    depth_fractional = float(depth_ppm) / 1_000_000.0
    model = compute_bls_model(
        time=time_arr,
        period=float(period),
        t0=float(t0),
        duration_hours=float(duration_hours),
        depth=float(depth_fractional),
    )

    finite_mask = np.isfinite(time_arr) & np.isfinite(flux_arr) & np.isfinite(flux_err_arr)
    n_finite = int(np.sum(finite_mask))
    if n_finite <= 4:
        raise ValueError("Insufficient finite points to compute diagnostics (need >4).")

    residuals = flux_arr[finite_mask] - model[finite_mask]
    rms = float(np.sqrt(np.mean(residuals**2)))
    chi2 = float(np.sum((residuals / flux_err_arr[finite_mask]) ** 2))
    reduced_chi2 = float(chi2 / (n_finite - 4))

    in_transit = model < (1.0 - depth_fractional / 2)
    n_in_transit = int(np.sum(in_transit & finite_mask))

    return {
        "period": float(period),
        "t0": float(t0),
        "duration_hours": float(duration_hours),
        "depth_ppm": float(depth_ppm),
        "rms_residual": float(rms),
        "chi2": float(chi2),
        "reduced_chi2": float(reduced_chi2),
        "n_in_transit": int(n_in_transit),
    }


__all__ = ["compute_transit_model"]

