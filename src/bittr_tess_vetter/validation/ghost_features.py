"""Ghost/scattered-light feature extraction (metrics-only).

This module provides pixel-level artifact features derived from TPF-like cubes:
- Difference image (out-of-transit - in-transit)
- Aperture contrast (in vs out of aperture depth ratio)
- Spatial uniformity (uniform vs localized dimming)
- PRF likeness (correlation with a Gaussian PRF proxy)
- Edge gradient / background trend (scattered light indicators)

It intentionally contains no mission-specific calibrations or curated lists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class GhostFeatures:
    """Per-target ghost/artifact features from pixel data."""

    tic_id: int
    sector: int
    in_aperture_depth: float
    out_aperture_depth: float
    aperture_contrast: float
    spatial_uniformity: float
    prf_likeness: float
    edge_gradient_strength: float
    background_trend: float
    ghost_like_score: float
    scattered_light_risk: float
    aperture_pixels_used: int
    stamp_shape: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tic_id": self.tic_id,
            "sector": self.sector,
            "in_aperture_depth": self.in_aperture_depth,
            "out_aperture_depth": self.out_aperture_depth,
            "aperture_contrast": self.aperture_contrast,
            "spatial_uniformity": self.spatial_uniformity,
            "prf_likeness": self.prf_likeness,
            "edge_gradient_strength": self.edge_gradient_strength,
            "background_trend": self.background_trend,
            "ghost_like_score": self.ghost_like_score,
            "scattered_light_risk": self.scattered_light_risk,
            "aperture_pixels_used": self.aperture_pixels_used,
            "stamp_shape": list(self.stamp_shape),
        }


def compute_aperture_contrast(
    diff_image: NDArray[np.floating],
    aperture_mask: NDArray[np.bool_],
    *,
    background_annulus: tuple[int, int] = (3, 6),
) -> tuple[float, float, float]:
    """Compute in-aperture vs out-of-aperture mean signal and ratio."""
    if diff_image.shape != aperture_mask.shape:
        raise ValueError(
            f"diff_image shape {diff_image.shape} must match aperture_mask shape {aperture_mask.shape}"
        )

    n_rows, n_cols = diff_image.shape
    center_row = n_rows // 2
    center_col = n_cols // 2

    row_coords, col_coords = np.ogrid[:n_rows, :n_cols]
    distances = np.sqrt((row_coords - center_row) ** 2 + (col_coords - center_col) ** 2)

    inner_radius, outer_radius = background_annulus
    annulus_mask = (distances >= inner_radius) & (distances < outer_radius)
    annulus_mask = annulus_mask & ~aperture_mask

    aperture_pixels = diff_image[aperture_mask]
    in_aperture_depth = 0.0 if len(aperture_pixels) == 0 else float(np.nanmean(aperture_pixels))

    annulus_pixels = diff_image[annulus_mask]
    if len(annulus_pixels) == 0:
        out_pixels = diff_image[~aperture_mask]
        out_aperture_depth = float(np.nanmean(out_pixels)) if len(out_pixels) > 0 else 0.0
    else:
        out_aperture_depth = float(np.nanmean(annulus_pixels))

    if abs(out_aperture_depth) < 1e-10:
        if abs(in_aperture_depth) < 1e-10:
            contrast_ratio = 1.0
        else:
            contrast_ratio = np.inf if in_aperture_depth > 0 else -np.inf
    else:
        contrast_ratio = in_aperture_depth / out_aperture_depth

    return in_aperture_depth, out_aperture_depth, float(contrast_ratio)


def compute_spatial_uniformity(
    diff_image: NDArray[np.floating],
    aperture_mask: NDArray[np.bool_],
) -> float:
    """Compute spatial uniformity of the signal within the aperture."""
    if diff_image.shape != aperture_mask.shape:
        raise ValueError(
            f"diff_image shape {diff_image.shape} must match aperture_mask shape {aperture_mask.shape}"
        )

    aperture_pixels = diff_image[aperture_mask]
    if len(aperture_pixels) < 2:
        return 0.5

    mean_val = float(np.nanmean(aperture_pixels))
    std_val = float(np.nanstd(aperture_pixels))

    if abs(mean_val) < 1e-10:
        image_range = float(np.nanmax(diff_image) - np.nanmin(diff_image))
        if image_range < 1e-10:
            return 1.0
        cv = std_val / image_range
    else:
        cv = std_val / abs(mean_val)

    uniformity = 1.0 / (1.0 + cv)
    return float(np.clip(uniformity, 0.0, 1.0))


def compute_prf_likeness(
    diff_image: NDArray[np.floating],
    expected_prf: NDArray[np.floating] | None = None,
    *,
    generate_gaussian_prf: bool = True,
    prf_sigma: float = 1.0,
) -> float:
    """Compute how PRF-like the signal is (0-1)."""
    n_rows, n_cols = diff_image.shape

    prf_to_use: NDArray[np.floating]
    if expected_prf is not None:
        prf_to_use = expected_prf
    elif generate_gaussian_prf:
        row_center = n_rows / 2
        col_center = n_cols / 2

        row_coords, col_coords = np.ogrid[:n_rows, :n_cols]
        distances_sq = (row_coords - row_center) ** 2 + (col_coords - col_center) ** 2
        generated_prf = np.exp(-distances_sq / (2 * prf_sigma**2))
        prf_to_use = generated_prf / np.sum(generated_prf)
    else:
        raise ValueError("expected_prf must be provided or generate_gaussian_prf must be True")

    if prf_to_use.shape != diff_image.shape:
        raise ValueError(
            f"PRF shape {prf_to_use.shape} must match diff_image shape {diff_image.shape}"
        )

    diff_norm = diff_image - np.nanmean(diff_image)
    prf_norm = prf_to_use - np.nanmean(prf_to_use)

    diff_std = float(np.nanstd(diff_norm))
    prf_std = float(np.nanstd(prf_norm))
    if diff_std < 1e-10 or prf_std < 1e-10:
        return 0.5

    correlation = float(np.nanmean(diff_norm * prf_norm) / (diff_std * prf_std))
    prf_likeness = (correlation + 1.0) / 2.0
    return float(np.clip(prf_likeness, 0.0, 1.0))


def compute_edge_gradient(diff_image: NDArray[np.floating]) -> tuple[float, float]:
    """Detect edge effects and gradients in the difference image."""
    n_rows, n_cols = diff_image.shape
    if n_rows < 2 or n_cols < 2:
        return 0.0, 0.0

    grad_row = np.zeros_like(diff_image)
    grad_row[1:-1, :] = (diff_image[2:, :] - diff_image[:-2, :]) / 2.0

    grad_col = np.zeros_like(diff_image)
    grad_col[:, 1:-1] = (diff_image[:, 2:] - diff_image[:, :-2]) / 2.0

    grad_magnitude = np.sqrt(grad_row**2 + grad_col**2)

    edge_width = max(1, min(n_rows, n_cols) // 4)

    edge_mask = np.zeros_like(diff_image, dtype=bool)
    edge_mask[:edge_width, :] = True
    edge_mask[-edge_width:, :] = True
    edge_mask[:, :edge_width] = True
    edge_mask[:, -edge_width:] = True

    center_mask = ~edge_mask

    edge_grad = float(np.nanmean(grad_magnitude[edge_mask])) if np.any(edge_mask) else 0.0
    center_grad = float(np.nanmean(grad_magnitude[center_mask])) if np.any(center_mask) else 0.0

    total_grad = edge_grad + center_grad
    edge_gradient_strength = 0.0 if total_grad < 1e-10 else edge_grad / total_grad

    row_coords, col_coords = np.meshgrid(
        np.arange(n_rows, dtype=float),
        np.arange(n_cols, dtype=float),
        indexing="ij",
    )

    flat_rows = row_coords.ravel()
    flat_cols = col_coords.ravel()
    flat_vals = diff_image.ravel()

    valid_mask = ~np.isnan(flat_vals)
    if int(np.sum(valid_mask)) < 3:
        return float(edge_gradient_strength), 0.0

    flat_rows = flat_rows[valid_mask]
    flat_cols = flat_cols[valid_mask]
    flat_vals = flat_vals[valid_mask]

    design_matrix = np.column_stack([flat_rows, flat_cols, np.ones_like(flat_rows)])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(design_matrix, flat_vals, rcond=None)
        a, b, _ = coeffs
        background_trend = float(np.sqrt(a**2 + b**2))

        image_std = float(np.nanstd(diff_image))
        if image_std > 1e-10:
            background_trend = background_trend / image_std
    except np.linalg.LinAlgError:
        background_trend = 0.0

    return float(edge_gradient_strength), float(background_trend)


def _compute_transit_mask(
    time: NDArray[np.floating],
    period: float,
    t0: float,
    duration_hours: float,
) -> NDArray[np.bool_]:
    duration_days = duration_hours / 24.0

    phase = ((time - t0) % period) / period
    phase = np.where(phase > 0.5, phase - 1.0, phase)

    half_duration_phase = (duration_days / 2) / period
    return np.abs(phase) <= half_duration_phase


def compute_difference_image(
    tpf_data: NDArray[np.floating],
    time: NDArray[np.floating],
    period: float,
    t0: float,
    duration_hours: float,
) -> NDArray[np.floating]:
    """Compute a difference image (out_of_transit - in_transit)."""
    in_transit_mask = _compute_transit_mask(time, period, t0, duration_hours)
    out_of_transit_mask = ~in_transit_mask

    n_in = int(np.sum(in_transit_mask))
    n_out = int(np.sum(out_of_transit_mask))

    if n_in == 0:
        raise ValueError("No in-transit data points found")
    if n_out == 0:
        raise ValueError("No out-of-transit data points found")

    in_transit_image = np.nanmedian(tpf_data[in_transit_mask], axis=0)
    out_of_transit_image = np.nanmedian(tpf_data[out_of_transit_mask], axis=0)
    return out_of_transit_image - in_transit_image


def _compute_ghost_like_score(
    *,
    spatial_uniformity: float,
    prf_likeness: float,
    aperture_contrast: float,
) -> float:
    if aperture_contrast <= 0:
        contrast_score = 1.0
    elif aperture_contrast >= 10:
        contrast_score = 0.0
    else:
        contrast_score = 1.0 - np.log10(max(aperture_contrast, 0.1)) / np.log10(10)
        contrast_score = float(np.clip(contrast_score, 0.0, 1.0))

    ghost_score = 0.4 * spatial_uniformity + 0.4 * (1.0 - prf_likeness) + 0.2 * contrast_score
    return float(np.clip(ghost_score, 0.0, 1.0))


def _compute_scattered_light_risk(
    *,
    edge_gradient_strength: float,
    background_trend: float,
) -> float:
    edge_contribution = edge_gradient_strength
    trend_contribution = 1.0 - np.exp(-2.0 * background_trend)
    trend_contribution = float(np.clip(trend_contribution, 0.0, 1.0))
    scattered_risk = 0.5 * edge_contribution + 0.5 * trend_contribution
    return float(np.clip(scattered_risk, 0.0, 1.0))


def compute_ghost_features(
    tpf_data: NDArray[np.floating],
    time: NDArray[np.floating],
    aperture_mask: NDArray[np.bool_],
    period: float,
    t0: float,
    duration_hours: float,
    *,
    tic_id: int = 0,
    sector: int = 0,
    background_annulus: tuple[int, int] = (3, 6),
    prf_sigma: float = 1.0,
) -> GhostFeatures:
    """Compute ghost/scattered-light features from a TPF-like cube."""
    if tpf_data.ndim != 3:
        raise ValueError(f"tpf_data must be 3D, got shape {tpf_data.shape}")

    n_cadences, n_rows, n_cols = tpf_data.shape

    if time.ndim != 1 or len(time) != n_cadences:
        raise ValueError(f"time must be 1D with length {n_cadences}")

    if aperture_mask.shape != (n_rows, n_cols):
        raise ValueError(
            f"aperture_mask shape {aperture_mask.shape} must match spatial dimensions ({n_rows}, {n_cols})"
        )

    diff_image = compute_difference_image(tpf_data, time, period, t0, duration_hours)

    in_aperture_depth, out_aperture_depth, aperture_contrast = compute_aperture_contrast(
        diff_image, aperture_mask, background_annulus=background_annulus
    )

    spatial_uniformity = compute_spatial_uniformity(diff_image, aperture_mask)
    prf_likeness = compute_prf_likeness(diff_image, prf_sigma=prf_sigma)
    edge_gradient_strength, background_trend = compute_edge_gradient(diff_image)

    ghost_like_score = _compute_ghost_like_score(
        spatial_uniformity=spatial_uniformity,
        prf_likeness=prf_likeness,
        aperture_contrast=aperture_contrast,
    )
    scattered_light_risk = _compute_scattered_light_risk(
        edge_gradient_strength=edge_gradient_strength,
        background_trend=background_trend,
    )

    return GhostFeatures(
        tic_id=int(tic_id),
        sector=int(sector),
        in_aperture_depth=float(in_aperture_depth),
        out_aperture_depth=float(out_aperture_depth),
        aperture_contrast=float(aperture_contrast),
        spatial_uniformity=float(spatial_uniformity),
        prf_likeness=float(prf_likeness),
        edge_gradient_strength=float(edge_gradient_strength),
        background_trend=float(background_trend),
        ghost_like_score=float(ghost_like_score),
        scattered_light_risk=float(scattered_light_risk),
        aperture_pixels_used=int(np.sum(aperture_mask)),
        stamp_shape=(int(n_rows), int(n_cols)),
    )


__all__ = [
    "GhostFeatures",
    "compute_aperture_contrast",
    "compute_difference_image",
    "compute_edge_gradient",
    "compute_ghost_features",
    "compute_prf_likeness",
    "compute_spatial_uniformity",
]

