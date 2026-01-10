"""Light curve stitching primitives (multi-sector normalization + merge).

This module provides a small, reusable primitive for combining multiple TESS
light curves (typically per-sector) into a single time series:

- Per-sector median normalization using only quality==0 cadences when possible
- Concatenation with gap preservation (no interpolation)
- Chronological sorting by time

Notes:
- Time is expected to be BTJD (same convention used throughout the bittr API).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from bittr_tess_vetter.domain.lightcurve import LightCurveData


def _infer_cadence_seconds(
    time: NDArray[np.floating[Any]],
    sector: NDArray[np.integer[Any]],
    *,
    default_seconds: float = 120.0,
) -> float:
    """Infer a representative cadence (seconds) for a stitched light curve.

    Uses within-sector adjacent cadence deltas only (after chronological sort) to
    avoid cross-sector boundary gaps contaminating the estimate.

    Notes:
    - If multiple sectors have different cadences (e.g., 20s vs 120s), this
      returns the median of within-sector cadence deltas across the full series,
      which approximates the dominant cadence.
    - Callers should treat cadence_seconds as informational; stitched series can
      be mixed-cadence.
    """
    if len(time) < 2:
        return float(default_seconds)

    dt = np.diff(time)
    same_sector = sector[1:] == sector[:-1]
    # Ignore large gaps (e.g., orbit downlinks) and cross-sector boundaries.
    short = dt[(dt > 0) & (dt < 1.0) & same_sector]
    if len(short) == 0:
        return float(default_seconds)
    return float(np.median(short) * 86400.0)


@dataclass(frozen=True)
class SectorDiagnostics:
    """Per-input diagnostics recorded during stitching."""

    sector: int
    n_cadences: int
    median_flux: float
    mad_flux: float
    normalization_factor: float
    normalization_warning: str | None = None
    quality_flags_summary: dict[str, int] = field(default_factory=dict)


@dataclass
class StitchedLC:
    """Stitched multi-sector light curve container."""

    time: NDArray[np.floating[Any]]
    flux: NDArray[np.floating[Any]]
    flux_err: NDArray[np.floating[Any]]
    sector: NDArray[np.integer[Any]]
    quality: NDArray[np.integer[Any]]
    per_sector_diagnostics: list[SectorDiagnostics]
    normalization_policy_version: str


def _compute_mad(data: NDArray[np.floating[Any]]) -> float:
    if not np.any(np.isfinite(data)):
        return float("nan")
    median = np.nanmedian(data)
    return float(np.nanmedian(np.abs(data - median)))


def _summarize_quality_flags(quality: NDArray[np.integer[Any]]) -> dict[str, int]:
    total = len(quality)
    good = int(np.sum(quality == 0))
    flagged = total - good
    return {"good": good, "flagged": flagged, "total": total}


def _validate_lightcurve_dict(lc: dict[str, Any], index: int) -> None:
    required_fields = ["time", "flux", "flux_err", "sector", "quality"]
    for field_name in required_fields:
        if field_name not in lc:
            raise ValueError(f"Light curve at index {index} missing required field: {field_name}")

    n = len(lc["time"])
    if len(lc["flux"]) != n:
        raise ValueError(
            f"Light curve at index {index}: flux length ({len(lc['flux'])}) "
            f"does not match time length ({n})"
        )
    if len(lc["flux_err"]) != n:
        raise ValueError(
            f"Light curve at index {index}: flux_err length ({len(lc['flux_err'])}) "
            f"does not match time length ({n})"
        )
    if len(lc["quality"]) != n:
        raise ValueError(
            f"Light curve at index {index}: quality length ({len(lc['quality'])}) "
            f"does not match time length ({n})"
        )


def _compute_normalization_factor_v1(
    flux: NDArray[np.floating[Any]],
    quality: NDArray[np.integer[Any]] | None = None,
) -> float:
    if quality is not None:
        good_mask = (quality == 0) & np.isfinite(flux)
        if np.any(good_mask):
            return float(np.median(flux[good_mask]))

    finite_mask = np.isfinite(flux)
    if np.any(finite_mask):
        return float(np.median(flux[finite_mask]))
    return float("nan")


def stitch_lightcurves(
    lc_list: list[dict[str, Any]],
    normalization_policy_version: str = "v1",
) -> StitchedLC:
    """Stitch multiple (typically per-sector) light curves into a single series.

    Args:
        lc_list: Each dict must include time/flux/flux_err/sector/quality.
            Time is expected to be BTJD.
        normalization_policy_version: Currently only "v1" is supported.
    """
    if not lc_list:
        raise ValueError("lc_list cannot be empty")

    if normalization_policy_version != "v1":
        raise ValueError(
            f"Unsupported normalization_policy_version: {normalization_policy_version}. "
            f"Only 'v1' is currently supported."
        )

    for i, lc in enumerate(lc_list):
        _validate_lightcurve_dict(lc, i)

    all_times: list[NDArray[np.floating[Any]]] = []
    all_fluxes: list[NDArray[np.floating[Any]]] = []
    all_flux_errs: list[NDArray[np.floating[Any]]] = []
    all_sectors: list[NDArray[np.integer[Any]]] = []
    all_qualities: list[NDArray[np.integer[Any]]] = []
    diagnostics: list[SectorDiagnostics] = []

    for lc in lc_list:
        time = np.asarray(lc["time"], dtype=np.float64)
        flux = np.asarray(lc["flux"], dtype=np.float64)
        flux_err = np.asarray(lc["flux_err"], dtype=np.float64)
        sector = int(lc["sector"])
        quality = np.asarray(lc["quality"], dtype=np.int32)

        n_cadences = len(time)

        finite_flux = flux[np.isfinite(flux)]
        median_flux = float(np.median(finite_flux)) if len(finite_flux) else float("nan")
        mad_flux = _compute_mad(flux)

        norm_factor = _compute_normalization_factor_v1(flux, quality)

        normalization_warning: str | None = None
        if np.isfinite(norm_factor) and norm_factor > 0:
            normalized_flux = flux / norm_factor
            normalized_flux_err = flux_err / norm_factor
        else:
            normalization_warning = (
                f"Invalid normalization factor ({norm_factor}); using unnormalized flux for this sector."
            )
            normalized_flux = flux
            normalized_flux_err = flux_err

        sector_arr = np.full(n_cadences, sector, dtype=np.int32)
        quality_summary = _summarize_quality_flags(quality)

        diagnostics.append(
            SectorDiagnostics(
                sector=sector,
                n_cadences=n_cadences,
                median_flux=median_flux,
                mad_flux=mad_flux,
                normalization_factor=norm_factor,
                normalization_warning=normalization_warning,
                quality_flags_summary=quality_summary,
            )
        )

        all_times.append(time)
        all_fluxes.append(normalized_flux)
        all_flux_errs.append(normalized_flux_err)
        all_sectors.append(sector_arr)
        all_qualities.append(quality)

    combined_time = np.concatenate(all_times)
    combined_flux = np.concatenate(all_fluxes)
    combined_flux_err = np.concatenate(all_flux_errs)
    combined_sector = np.concatenate(all_sectors)
    combined_quality = np.concatenate(all_qualities)

    sort_idx = np.argsort(combined_time)
    combined_time = combined_time[sort_idx]
    combined_flux = combined_flux[sort_idx]
    combined_flux_err = combined_flux_err[sort_idx]
    combined_sector = combined_sector[sort_idx]
    combined_quality = combined_quality[sort_idx]

    diagnostics.sort(key=lambda d: d.sector)

    return StitchedLC(
        time=combined_time,
        flux=combined_flux,
        flux_err=combined_flux_err,
        sector=combined_sector,
        quality=combined_quality,
        per_sector_diagnostics=diagnostics,
        normalization_policy_version=normalization_policy_version,
    )

def stitch_lightcurve_data(
    lightcurves: list[LightCurveData],
    *,
    tic_id: int,
    normalization_policy_version: str = "v1",
    sector: int = -1,
) -> tuple[LightCurveData, StitchedLC]:
    """Stitch cached `LightCurveData` objects into a single `LightCurveData`.

    This helper is host-facing: it converts `LightCurveData` -> dict inputs for
    `stitch_lightcurves`, then converts the stitched output back into an immutable
    `LightCurveData` with a conservative `valid_mask` and inferred cadence.
    """
    if tic_id <= 0:
        raise ValueError(f"tic_id must be positive, got {tic_id}")
    if len(lightcurves) < 2:
        raise ValueError("stitch_lightcurve_data requires at least 2 input light curves")

    lc_list: list[dict[str, Any]] = []
    for lc in lightcurves:
        lc_list.append(
            {
                "time": lc.time,
                "flux": lc.flux,
                "flux_err": lc.flux_err,
                "sector": int(lc.sector),
                "quality": lc.quality,
            }
        )

    stitched = stitch_lightcurves(lc_list, normalization_policy_version=normalization_policy_version)

    valid_mask = (stitched.quality == 0) & np.isfinite(stitched.flux)

    cadence_seconds = _infer_cadence_seconds(stitched.time, stitched.sector, default_seconds=120.0)

    stitched_lc_data = LightCurveData(
        time=np.asarray(stitched.time, dtype=np.float64),
        flux=np.asarray(stitched.flux, dtype=np.float64),
        flux_err=np.asarray(stitched.flux_err, dtype=np.float64),
        quality=np.asarray(stitched.quality, dtype=np.int32),
        valid_mask=np.asarray(valid_mask, dtype=np.bool_),
        tic_id=int(tic_id),
        sector=int(sector),
        cadence_seconds=float(cadence_seconds),
    )

    return stitched_lc_data, stitched


__all__ = ["SectorDiagnostics", "StitchedLC", "stitch_lightcurves", "stitch_lightcurve_data"]
