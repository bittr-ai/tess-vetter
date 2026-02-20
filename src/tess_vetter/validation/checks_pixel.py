"""Pixel-level vetting computations (metrics-only).

This module provides V08–V10 measurements that require TPF-like data:
- V08: centroid shift metrics (in vs out of transit)
- V09: per-pixel depth map + concentration metrics
- V10: aperture dependence metrics (depth vs aperture radius)

All results are metrics-only: `passed=None` and `details["_metrics_only"]=True`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tess_vetter.domain.detection import TransitCandidate, VetterCheckResult
from tess_vetter.pixel.aperture import TransitParams as ApertureTransitParams
from tess_vetter.pixel.aperture import compute_aperture_dependence
from tess_vetter.pixel.centroid import TransitParams as CentroidTransitParams
from tess_vetter.pixel.centroid import compute_centroid_shift
from tess_vetter.validation.base import (
    get_in_transit_mask,
    get_out_of_transit_mask,
)


def _metrics_result(
    *,
    check_id: str,
    name: str,
    confidence: float,
    details: dict[str, Any],
) -> VetterCheckResult:
    details = dict(details)
    details["_metrics_only"] = True
    return VetterCheckResult(
        id=check_id,
        name=name,
        passed=None,
        confidence=float(max(0.0, min(1.0, confidence))),
        details=details,
    )


# =============================================================================
# V08: Centroid shift (wrapper around pixel.centroid)
# =============================================================================


def check_centroid_shift_with_tpf(
    *,
    tpf_data: NDArray[np.floating[Any]],
    time: NDArray[np.floating[Any]],
    candidate: TransitCandidate,
    centroid_method: str = "median",
    significance_method: str = "bootstrap",
    n_bootstrap: int = 1000,
    bootstrap_seed: int | None = None,
    window_policy_version: str = "v1",
    outlier_sigma: float = 3.0,
) -> VetterCheckResult:
    params = CentroidTransitParams(
        period=float(candidate.period),
        t0=float(candidate.t0),
        duration=float(candidate.duration_hours),
    )
    result = compute_centroid_shift(
        tpf_data=tpf_data,
        time=time,
        transit_params=params,
        window_policy_version=window_policy_version,
        significance_method=significance_method,  # type: ignore[arg-type]
        n_bootstrap=int(n_bootstrap),
        bootstrap_seed=bootstrap_seed,
        centroid_method=centroid_method,  # type: ignore[arg-type]
        outlier_sigma=float(outlier_sigma),
    )

    n_in = int(result.n_in_transit_cadences)
    n_out = int(result.n_out_transit_cadences)
    confidence = min(1.0, (n_in / 10.0)) * min(1.0, (n_out / 50.0))

    # Compute reference image (out-of-transit median) for plotting
    # Apply same cadence mask as centroid computation
    time_arr = np.asarray(time, dtype=np.float64)
    tpf_arr = np.asarray(tpf_data, dtype=np.float64)
    finite_time = np.isfinite(time_arr)
    frame_has_finite = np.any(np.isfinite(tpf_arr), axis=(1, 2))
    cadence_mask = finite_time & frame_has_finite
    tpf_filtered = tpf_arr[cadence_mask]
    time_filtered = time_arr[cadence_mask]

    # Compute out-of-transit mask
    out_mask = get_out_of_transit_mask(
        time_filtered,
        float(candidate.period),
        float(candidate.t0),
        float(candidate.duration_hours),
        buffer_factor=1.5,
    )
    if np.any(out_mask):
        reference_image = np.nanmedian(tpf_filtered[out_mask], axis=0)
    else:
        reference_image = np.nanmedian(tpf_filtered, axis=0)

    # Cap image to 21x21 for plot_data
    max_size = 21
    n_rows, n_cols = reference_image.shape
    if n_rows > max_size or n_cols > max_size:
        row_start = (n_rows - max_size) // 2 if n_rows > max_size else 0
        col_start = (n_cols - max_size) // 2 if n_cols > max_size else 0
        row_end = row_start + min(n_rows, max_size)
        col_end = col_start + min(n_cols, max_size)
        reference_image = reference_image[row_start:row_end, col_start:col_end]
    else:
        row_start, col_start = 0, 0

    # Target pixel is center of the TPF (adjust for any cropping)
    target_col = int(n_cols // 2) - col_start
    target_row = int(n_rows // 2) - row_start

    # Centroids: (x, y) = (col, row) from compute_centroid_shift
    # Adjust for any cropping
    in_col = float(result.in_transit_centroid[0]) - col_start
    in_row = float(result.in_transit_centroid[1]) - row_start
    out_col = float(result.out_of_transit_centroid[0]) - col_start
    out_row = float(result.out_of_transit_centroid[1]) - row_start

    # Build plot_data
    plot_data: dict[str, Any] = {
        "version": 1,
        "reference_image": reference_image.astype(np.float32).tolist(),
        "in_centroid_col": in_col,
        "in_centroid_row": in_row,
        "out_centroid_col": out_col,
        "out_centroid_row": out_row,
        "target_col": target_col,
        "target_row": target_row,
    }

    return _metrics_result(
        check_id="V08",
        name="centroid_shift",
        confidence=confidence,
        details={
            "status": "ok",
            "centroid_shift_pixels": float(result.centroid_shift_pixels),
            "centroid_shift_arcsec": float(result.centroid_shift_arcsec),
            "significance_sigma": float(result.significance_sigma),
            "shift_uncertainty_pixels": float(result.shift_uncertainty_pixels),
            "shift_ci_lower_pixels": float(result.shift_ci_lower_pixels),
            "shift_ci_upper_pixels": float(result.shift_ci_upper_pixels),
            "in_transit_centroid": tuple(result.in_transit_centroid),
            "out_of_transit_centroid": tuple(result.out_of_transit_centroid),
            "in_transit_centroid_se": tuple(result.in_transit_centroid_se),
            "out_of_transit_centroid_se": tuple(result.out_of_transit_centroid_se),
            "n_in_transit_cadences": n_in,
            "n_out_of_transit_cadences": n_out,
            "centroid_method": result.centroid_method,
            "significance_method": result.significance_method,
            "n_bootstrap": int(result.n_bootstrap),
            "saturation_risk": bool(result.saturation_risk),
            "max_flux_fraction": float(result.max_flux_fraction),
            "n_outliers_rejected": int(result.n_outliers_rejected),
            "warnings": list(result.warnings),
            "tpf_shape": list(tpf_data.shape),
            "plot_data": plot_data,
        },
    )


# =============================================================================
# V09: Pixel-level depth map + concentration metrics
# =============================================================================


@dataclass(frozen=True)
class PixelDepthMapMetrics:
    depth_map_ppm: NDArray[np.floating[Any]]
    max_depth_pixel: tuple[int, int]
    max_depth_ppm: float
    target_pixel: tuple[int, int]
    target_depth_ppm: float
    concentration_ratio: float
    distance_to_target_pixels: float


def compute_pixel_level_depths_ppm(
    *,
    tpf_data: NDArray[np.floating[Any]],
    time: NDArray[np.floating[Any]],
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
) -> NDArray[np.floating[Any]]:
    """Compute per-pixel transit depths in ppm using simple in/out masks."""
    if tpf_data.ndim != 3:
        raise ValueError(f"tpf_data must be 3D (time, rows, cols), got {tpf_data.shape}")
    if time.shape[0] != tpf_data.shape[0]:
        raise ValueError("time length must match tpf_data first axis")

    time = np.asarray(time, dtype=np.float64)
    tpf_data = np.asarray(tpf_data, dtype=np.float64)

    finite_time = np.isfinite(time)
    frame_has_finite = np.any(np.isfinite(tpf_data), axis=(1, 2))
    cadence_mask = finite_time & frame_has_finite

    time = time[cadence_mask]
    tpf_data = tpf_data[cadence_mask]

    in_mask = get_in_transit_mask(time, period_days, t0_btjd, duration_hours)
    out_mask = get_out_of_transit_mask(
        time, period_days, t0_btjd, duration_hours, buffer_factor=3.0
    )

    in_flux = tpf_data[in_mask]
    out_flux = tpf_data[out_mask]
    if in_flux.size == 0 or out_flux.size == 0:
        raise ValueError("insufficient in/out-of-transit cadences for pixel depth map")

    out_median = np.nanmedian(out_flux, axis=0)
    in_median = np.nanmedian(in_flux, axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        depth_frac = (out_median - in_median) / out_median
    depth_ppm = depth_frac * 1e6
    depth_ppm[~np.isfinite(depth_ppm)] = np.nan
    return depth_ppm


def _default_target_pixel(depth_map_ppm: NDArray[np.floating[Any]]) -> tuple[int, int]:
    n_rows, n_cols = depth_map_ppm.shape
    return (n_rows // 2, n_cols // 2)


def compute_pixel_depth_map_metrics(
    *,
    depth_map_ppm: NDArray[np.floating[Any]],
    target_pixel: tuple[int, int] | None = None,
) -> PixelDepthMapMetrics:
    if depth_map_ppm.ndim != 2:
        raise ValueError("depth_map_ppm must be 2D")

    target = target_pixel or _default_target_pixel(depth_map_ppm)
    target_row, target_col = target

    valid = np.isfinite(depth_map_ppm) & (depth_map_ppm > 0)
    if not np.any(valid):
        return PixelDepthMapMetrics(
            depth_map_ppm=depth_map_ppm,
            max_depth_pixel=(0, 0),
            max_depth_ppm=0.0,
            target_pixel=target,
            target_depth_ppm=0.0,
            concentration_ratio=0.0,
            distance_to_target_pixels=float("nan"),
        )

    masked = np.where(valid, depth_map_ppm, -np.inf)
    max_idx = np.unravel_index(int(np.argmax(masked)), depth_map_ppm.shape)
    max_pixel = (int(max_idx[0]), int(max_idx[1]))
    max_depth_ppm = float(depth_map_ppm[max_pixel])

    if (
        0 <= target_row < depth_map_ppm.shape[0]
        and 0 <= target_col < depth_map_ppm.shape[1]
        and np.isfinite(depth_map_ppm[target_row, target_col])
    ):
        target_depth_ppm = float(depth_map_ppm[target_row, target_col])
    else:
        target_depth_ppm = 0.0

    concentration_ratio = target_depth_ppm / max_depth_ppm if max_depth_ppm > 0 else 0.0
    distance = float(np.hypot(max_pixel[0] - target_row, max_pixel[1] - target_col))

    return PixelDepthMapMetrics(
        depth_map_ppm=depth_map_ppm,
        max_depth_pixel=max_pixel,
        max_depth_ppm=max_depth_ppm,
        target_pixel=target,
        target_depth_ppm=target_depth_ppm,
        concentration_ratio=float(concentration_ratio),
        distance_to_target_pixels=distance,
    )


def check_pixel_level_lc_with_tpf(
    *,
    tpf_data: NDArray[np.floating[Any]],
    time: NDArray[np.floating[Any]],
    candidate: TransitCandidate,
    target_pixel: tuple[int, int] | None = None,
) -> VetterCheckResult:
    def _edge_distance(pixel: tuple[int, int], shape: tuple[int, int]) -> int:
        r, c = int(pixel[0]), int(pixel[1])
        n_rows, n_cols = int(shape[0]), int(shape[1])
        return int(min(r, c, (n_rows - 1) - r, (n_cols - 1) - c))

    try:
        depth_map = compute_pixel_level_depths_ppm(
            tpf_data=tpf_data,
            time=time,
            period_days=float(candidate.period),
            t0_btjd=float(candidate.t0),
            duration_hours=float(candidate.duration_hours),
        )
        metrics = compute_pixel_depth_map_metrics(
            depth_map_ppm=depth_map, target_pixel=target_pixel
        )
        confidence = 0.7
        warnings: list[str] = []
        if not np.isfinite(metrics.distance_to_target_pixels):
            confidence = 0.2
            warnings.append("NO_VALID_PIXEL_DEPTHS")

        # Reliability diagnostics for bright/saturated/edge-dominated stamps.
        n_rows, n_cols = depth_map.shape
        max_edge_dist = _edge_distance(metrics.max_depth_pixel, (n_rows, n_cols))
        target_edge_dist = _edge_distance(metrics.target_pixel, (n_rows, n_cols))

        max_at_edge = max_edge_dist <= 1
        target_depth_nonpositive = not (metrics.target_depth_ppm > 0)
        max_depth_nonpositive = not (metrics.max_depth_ppm > 0)

        if max_at_edge:
            warnings.append("DIFFIMG_MAX_AT_EDGE")
        if target_depth_nonpositive:
            warnings.append("DIFFIMG_TARGET_DEPTH_NONPOSITIVE")
        if max_depth_nonpositive:
            warnings.append("DIFFIMG_MAX_DEPTH_NONPOSITIVE")

        localization_reliable = (
            np.isfinite(metrics.distance_to_target_pixels)
            and not max_at_edge
            and not target_depth_nonpositive
            and not max_depth_nonpositive
        )
        if not localization_reliable:
            warnings.append("DIFFIMG_UNRELIABLE")
    except ValueError as e:
        return _metrics_result(
            check_id="V09",
            name="pixel_level_lc",
            confidence=0.0,
            details={
                "status": "insufficient_data",
                "error": str(e),
                "tpf_shape": list(tpf_data.shape),
            },
        )
    except Exception as e:
        return _metrics_result(
            check_id="V09",
            name="pixel_level_lc",
            confidence=0.0,
            details={"status": "error", "error": str(e), "tpf_shape": list(tpf_data.shape)},
        )

    # Build difference image (out-of-transit median - in-transit median) for plotting
    time_arr = np.asarray(time, dtype=np.float64)
    tpf_arr = np.asarray(tpf_data, dtype=np.float64)
    finite_time = np.isfinite(time_arr)
    frame_has_finite = np.any(np.isfinite(tpf_arr), axis=(1, 2))
    cadence_mask_v09 = finite_time & frame_has_finite
    tpf_filtered = tpf_arr[cadence_mask_v09]
    time_filtered = time_arr[cadence_mask_v09]

    in_mask = get_in_transit_mask(
        time_filtered,
        float(candidate.period),
        float(candidate.t0),
        float(candidate.duration_hours),
    )
    out_mask = get_out_of_transit_mask(
        time_filtered,
        float(candidate.period),
        float(candidate.t0),
        float(candidate.duration_hours),
        buffer_factor=3.0,
    )

    if np.any(in_mask) and np.any(out_mask):
        in_median = np.nanmedian(tpf_filtered[in_mask], axis=0)
        out_median = np.nanmedian(tpf_filtered[out_mask], axis=0)
        difference_image = out_median - in_median
    else:
        difference_image = np.zeros_like(depth_map)

    # Cap images to 21x21 for plot_data
    max_size = 21
    n_rows_orig, n_cols_orig = depth_map.shape
    if n_rows_orig > max_size or n_cols_orig > max_size:
        row_start = (n_rows_orig - max_size) // 2 if n_rows_orig > max_size else 0
        col_start = (n_cols_orig - max_size) // 2 if n_cols_orig > max_size else 0
        row_end = row_start + min(n_rows_orig, max_size)
        col_end = col_start + min(n_cols_orig, max_size)
        depth_map_cropped = depth_map[row_start:row_end, col_start:col_end]
        difference_image_cropped = difference_image[row_start:row_end, col_start:col_end]
    else:
        row_start, col_start = 0, 0
        depth_map_cropped = depth_map
        difference_image_cropped = difference_image

    # Adjust pixel coordinates for cropping
    target_pixel_adj = [
        int(metrics.target_pixel[0]) - row_start,
        int(metrics.target_pixel[1]) - col_start,
    ]
    max_depth_pixel_adj = [
        int(metrics.max_depth_pixel[0]) - row_start,
        int(metrics.max_depth_pixel[1]) - col_start,
    ]

    # Build plot_data for V09
    plot_data_v09: dict[str, Any] = {
        "version": 1,
        "difference_image": difference_image_cropped.astype(np.float32).tolist(),
        "depth_map_ppm": depth_map_cropped.astype(np.float32).tolist(),
        "target_pixel": target_pixel_adj,  # [row, col]
        "max_depth_pixel": max_depth_pixel_adj,  # [row, col]
    }

    return _metrics_result(
        check_id="V09",
        name="pixel_level_lc",
        confidence=confidence,
        details={
            "status": "ok",
            "max_depth_pixel": metrics.max_depth_pixel,
            "max_depth_ppm": metrics.max_depth_ppm,
            "max_depth_ppm_abs": float(abs(metrics.max_depth_ppm)),
            "target_pixel": metrics.target_pixel,
            "target_depth_ppm": metrics.target_depth_ppm,
            "target_depth_ppm_abs": float(abs(metrics.target_depth_ppm)),
            "concentration_ratio": metrics.concentration_ratio,
            "concentration_ratio_abs": float(
                abs(metrics.target_depth_ppm) / abs(metrics.max_depth_ppm)
            )
            if metrics.max_depth_ppm != 0
            else 0.0,
            "distance_to_target_pixels": metrics.distance_to_target_pixels,
            "max_depth_pixel_edge_distance": int(max_edge_dist),
            "target_pixel_edge_distance": int(target_edge_dist),
            "localization_reliable": bool(localization_reliable),
            "tpf_shape": list(tpf_data.shape),
            "n_cadences": int(time.shape[0]),
            "warnings": warnings,
            "plot_data": plot_data_v09,
        },
    )


# =============================================================================
# V10: Aperture dependence (wrapper around pixel.aperture)
# =============================================================================


def check_aperture_dependence_with_tpf(
    *,
    tpf_data: NDArray[np.floating[Any]],
    time: NDArray[np.floating[Any]],
    candidate: TransitCandidate,
    aperture_radii_px: list[float] | None = None,
    center_row_col: tuple[float, float] | None = None,
) -> VetterCheckResult:
    params = ApertureTransitParams(
        period=float(candidate.period),
        t0=float(candidate.t0),
        duration=float(candidate.duration_hours) / 24.0,
    )
    try:
        result = compute_aperture_dependence(
            tpf_data=tpf_data,
            time=time,
            transit_params=params,
            aperture_radii=aperture_radii_px,
            center=center_row_col,
        )
    except ValueError as e:
        return _metrics_result(
            check_id="V10",
            name="aperture_dependence",
            confidence=0.0,
            details={
                "status": "insufficient_data",
                "error": str(e),
                "tpf_shape": list(tpf_data.shape),
            },
        )
    except Exception as e:
        return _metrics_result(
            check_id="V10",
            name="aperture_dependence",
            confidence=0.0,
            details={"status": "error", "error": str(e), "tpf_shape": list(tpf_data.shape)},
        )

    n_in = int(result.n_in_transit_cadences)
    n_out = int(result.n_out_of_transit_cadences)
    confidence = min(1.0, (n_in / 10.0)) * min(1.0, (n_out / 50.0))

    # Minimal additional diagnostics (useful for bright targets where apertures
    # can behave pathologically).
    depths_map = {float(k): float(v) for k, v in result.depths_by_aperture.items()}
    radii = sorted(depths_map.keys())
    depth_small = depths_map[radii[0]] if radii else float("nan")
    depth_large = depths_map[radii[-1]] if radii else float("nan")
    sign_flip = bool(np.isfinite(depth_small) and np.isfinite(depth_large) and depth_small * depth_large < 0)

    # Build plot_data for V10
    # Extract sorted radii and corresponding depths for plotting
    aperture_radii_list = [float(r) for r in radii]
    depths_ppm_list = [float(depths_map[r]) for r in radii]
    # Compute depth errors as sqrt(variance) / sqrt(n_apertures) as a simple estimate
    # This is a rough approximation; real errors would need per-aperture bootstrap
    depth_variance = float(result.depth_variance)
    n_apertures = len(radii)
    if n_apertures > 1 and depth_variance > 0:
        # Use coefficient of variation as a proxy for relative error
        mean_depth = float(np.mean(depths_ppm_list))
        if abs(mean_depth) > 0:
            cv = float(np.sqrt(depth_variance)) / abs(mean_depth)
            depth_errs_ppm_list = [abs(d) * cv for d in depths_ppm_list]
        else:
            depth_errs_ppm_list = [float(np.sqrt(depth_variance))] * n_apertures
    else:
        # No variance info, use zero errors
        depth_errs_ppm_list = [0.0] * n_apertures

    plot_data_v10: dict[str, Any] = {
        "version": 1,
        "aperture_radii_px": aperture_radii_list,
        "depths_ppm": depths_ppm_list,
        "depth_errs_ppm": depth_errs_ppm_list,
    }

    return _metrics_result(
        check_id="V10",
        name="aperture_dependence",
        confidence=confidence,
        details={
            "status": "ok",
            "stability_metric": float(result.stability_metric),
            "depths_by_aperture_ppm": {
                str(k): float(v) for k, v in result.depths_by_aperture.items()
            },
            "depth_ppm_aperture_min": float(depth_small),
            "depth_ppm_aperture_max": float(depth_large),
            "aperture_depth_sign_flip": bool(sign_flip),
            "depth_variance_ppm2": float(result.depth_variance),
            "recommended_aperture_pixels": float(result.recommended_aperture),
            "n_in_transit_cadences": n_in,
            "n_out_of_transit_cadences": n_out,
            "n_transit_epochs": int(result.n_transit_epochs),
            "baseline_mode": result.baseline_mode,
            "local_baseline_window_days": float(result.local_baseline_window_days),
            "background_mode": result.background_mode,
            "background_annulus_radii_pixels": tuple(result.background_annulus_radii_pixels),
            "n_background_pixels": int(result.n_background_pixels),
            "drift_fraction_recommended": result.drift_fraction_recommended,
            "flags": list(result.flags),
            "notes": dict(result.notes),
            "tpf_shape": list(tpf_data.shape),
            "plot_data": plot_data_v10,
        },
    )


__all__ = [
    "check_centroid_shift_with_tpf",
    "check_pixel_level_lc_with_tpf",
    "check_aperture_dependence_with_tpf",
    "compute_pixel_level_depths_ppm",
    "compute_pixel_depth_map_metrics",
    "PixelDepthMapMetrics",
]
