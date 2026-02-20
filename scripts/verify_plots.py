#!/usr/bin/env python3
"""Generate sample plots for visual verification.

This script creates all plotting functions with mock data and saves
the output to a verification directory for visual inspection.

Usage:
    uv run --extra plotting -- python scripts/verify_plots.py

    # Custom output directory
    uv run --extra plotting -- python scripts/verify_plots.py --out-dir working_docs/image_support/verification/verification_plots
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure matplotlib uses non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

# Add src to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tess_vetter.validation.result_schema import CheckResult


def ok_result(id: str, name: str, metrics: dict, raw: dict | None = None) -> CheckResult:
    """Create a mock CheckResult."""
    return CheckResult(
        id=id,
        name=name,
        status="ok",
        metrics=metrics,
        raw=raw,
    )


def create_output_dir(out_dir: str | Path | None) -> Path:
    """Create output directory for verification plots."""
    if out_dir is None:
        output_dir = (
            Path(__file__).parent.parent
            / "working_docs"
            / "image_support"
            / "verification"
            / "verification_plots"
        )
    else:
        output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def generate_v01_plot(output_dir: Path) -> None:
    """Generate V01 odd-even depth plot."""
    from tess_vetter.plotting import plot_odd_even

    result = ok_result(
        id="V01",
        name="Odd-Even Depth",
        metrics={"sigma_diff": 1.5, "depth_odd_ppm": 250.0, "depth_even_ppm": 245.0},
        raw={
            "plot_data": {
                "version": 1,
                "odd_epochs": [1, 3, 5, 7, 9],
                "odd_depths_ppm": [248.0, 252.0, 249.0, 251.0, 250.0],
                "odd_errs_ppm": [15.0, 14.0, 16.0, 15.0, 14.0],
                "even_epochs": [2, 4, 6, 8, 10],
                "even_depths_ppm": [244.0, 246.0, 245.0, 245.0, 246.0],
                "even_errs_ppm": [14.0, 15.0, 14.0, 15.0, 14.0],
                "mean_odd_ppm": 250.0,
                "mean_even_ppm": 245.2,
            }
        },
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_odd_even(result, ax=ax)
    fig.savefig(output_dir / "v01_odd_even.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ V01 odd-even plot saved")


def generate_v02_plot(output_dir: Path) -> None:
    """Generate V02 secondary eclipse plot."""
    from tess_vetter.plotting import plot_secondary_eclipse

    np.random.seed(42)
    phase = np.linspace(0, 1, 500)
    flux = 1.0 + np.random.normal(0, 0.001, 500)
    # Add primary transit
    flux[np.abs(phase - 0.0) < 0.02] -= 0.002
    flux[np.abs(phase - 1.0) < 0.02] -= 0.002

    result = ok_result(
        id="V02",
        name="Secondary Eclipse",
        metrics={"secondary_depth_ppm": 50.0, "secondary_snr": 2.1},
        raw={
            "plot_data": {
                "version": 1,
                "phase": phase.tolist(),
                "flux": flux.tolist(),
                "flux_err": (np.ones(500) * 0.001).tolist(),
                "secondary_window": [0.4, 0.6],
                "primary_window": [-0.05, 0.05],
                "secondary_depth_ppm": 50.0,
            }
        },
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_secondary_eclipse(result, ax=ax)
    fig.savefig(output_dir / "v02_secondary_eclipse.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ V02 secondary eclipse plot saved")


def generate_v04_plot(output_dir: Path) -> None:
    """Generate V04 depth stability plot."""
    from tess_vetter.plotting import plot_depth_stability

    np.random.seed(42)
    epochs = np.arange(1, 11)
    depths = 250 + np.random.normal(0, 15, 10)

    result = ok_result(
        id="V04",
        name="Depth Stability",
        metrics={"chi2_reduced": 1.2, "depth_scatter_ppm": 15.0},
        raw={
            "plot_data": {
                "version": 1,
                "epoch_times_btjd": (2459000 + epochs * 14.24).tolist(),
                "depths_ppm": depths.tolist(),
                "depth_errs_ppm": (np.ones(10) * 12.0).tolist(),
                "mean_depth_ppm": 250.0,
                "expected_scatter_ppm": 12.0,
                "dominating_epoch_idx": None,
            }
        },
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_depth_stability(result, ax=ax)
    fig.savefig(output_dir / "v04_depth_stability.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ V04 depth stability plot saved")


def generate_v05_plot(output_dir: Path) -> None:
    """Generate V05 V-shape plot."""
    from tess_vetter.plotting import plot_v_shape

    # Create binned transit shape
    phase = np.linspace(-0.05, 0.05, 20)
    # U-shaped transit
    flux = np.ones(20)
    in_transit = np.abs(phase) < 0.02
    flux[in_transit] = 1.0 - 0.002 * (1 - (phase[in_transit] / 0.02) ** 2)

    # Trapezoid model
    trap_phase = np.linspace(-0.05, 0.05, 100)
    trap_flux = np.ones(100)
    trap_flux[np.abs(trap_phase) < 0.015] = 0.998

    result = ok_result(
        id="V05",
        name="V-Shape",
        metrics={"tflat_ttotal_ratio": 0.6},
        raw={
            "plot_data": {
                "version": 1,
                "binned_phase": phase.tolist(),
                "binned_flux": flux.tolist(),
                "binned_flux_err": (np.ones(20) * 0.0003).tolist(),
                "trapezoid_phase": trap_phase.tolist(),
                "trapezoid_flux": trap_flux.tolist(),
                "t_flat_hours": 2.4,
                "t_total_hours": 4.0,
            }
        },
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_v_shape(result, ax=ax)
    fig.savefig(output_dir / "v05_v_shape.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ V05 V-shape plot saved")


def generate_v08_plot(output_dir: Path) -> None:
    """Generate V08 centroid shift plot."""
    from tess_vetter.plotting import plot_centroid_shift

    # Create mock TPF image
    np.random.seed(42)
    image = np.random.poisson(1000, (11, 11)).astype(float)
    # Add PSF-like center
    for i in range(11):
        for j in range(11):
            dist = np.sqrt((i - 5) ** 2 + (j - 5) ** 2)
            image[i, j] += 5000 * np.exp(-dist ** 2 / 4)

    result = ok_result(
        id="V08",
        name="Centroid Shift",
        metrics={"centroid_shift_arcsec": 0.5, "shift_significance_sigma": 2.1},
        raw={
            "plot_data": {
                "version": 1,
                "reference_image": image.tolist(),
                "in_centroid_col": 5.2,
                "in_centroid_row": 5.3,
                "out_centroid_col": 5.0,
                "out_centroid_row": 5.0,
                "target_col": 5,
                "target_row": 5,
            }
        },
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_centroid_shift(result, ax=ax)
    fig.savefig(output_dir / "v08_centroid_shift.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ V08 centroid shift plot saved")


def generate_v09_plot(output_dir: Path) -> None:
    """Generate V09 difference image plot."""
    from tess_vetter.plotting import plot_difference_image

    # Create mock difference image
    np.random.seed(42)
    diff_image = np.random.normal(0, 10, (11, 11))
    # Add transit signal at target
    diff_image[5, 5] = -200  # Negative = flux decrease during transit
    diff_image[4:7, 4:7] -= 50

    result = ok_result(
        id="V09",
        name="Difference Image",
        metrics={"concentration_ratio": 0.8},
        raw={
            "plot_data": {
                "version": 1,
                "difference_image": diff_image.tolist(),
                "depth_map_ppm": (diff_image * -1).tolist(),  # Positive depths
                "target_pixel": [5, 5],
                "max_depth_pixel": [5, 5],
            }
        },
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_difference_image(result, ax=ax)
    fig.savefig(output_dir / "v09_difference_image.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ V09 difference image plot saved")


def generate_v21_plot(output_dir: Path) -> None:
    """Generate V21 sector consistency plot."""
    from tess_vetter.plotting import plot_sector_consistency

    result = ok_result(
        id="V21",
        name="Sector Consistency",
        metrics={"chi2_p_value": 0.45},
        raw={
            "plot_data": {
                "version": 1,
                "sectors": [55, 75, 82, 83],
                "depths_ppm": [252.0, 248.0, 255.0, 250.0],
                "depth_errs_ppm": [15.0, 14.0, 16.0, 15.0],
                "weighted_mean_ppm": 251.0,
                "outlier_sectors": [],
            }
        },
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_sector_consistency(result, ax=ax)
    fig.savefig(output_dir / "v21_sector_consistency.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ V21 sector consistency plot saved")


def generate_v03_plot(output_dir: Path) -> None:
    """Generate V03 duration consistency plot."""
    from tess_vetter.plotting import plot_duration_consistency

    result = ok_result(
        id="V03",
        name="Duration Consistency",
        metrics={"duration_ratio": 0.95},
        raw={
            "plot_data": {
                "version": 1,
                "observed_hours": 3.8,
                "expected_hours": 4.0,
                "expected_hours_err": 0.5,
                "duration_ratio": 0.95,
            }
        },
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    plot_duration_consistency(result, ax=ax)
    fig.savefig(output_dir / "v03_duration_consistency.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ V03 duration consistency plot saved")


def generate_v06_plot(output_dir: Path) -> None:
    """Generate V06 nearby EBs plot."""
    from tess_vetter.plotting import plot_nearby_ebs

    result = ok_result(
        id="V06",
        name="Nearby EBs",
        metrics={"n_matches": 2},
        raw={
            "plot_data": {
                "version": 1,
                "target_ra": 120.0,
                "target_dec": -45.0,
                "matches": [
                    {"ra": 120.002, "dec": -44.998, "sep_arcsec": 15.0, "period_days": 1.23},
                    {"ra": 119.997, "dec": -45.003, "sep_arcsec": 22.5, "period_days": 2.56},
                ],
                "search_radius_arcsec": 42.0,
            }
        },
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_nearby_ebs(result, ax=ax)
    fig.savefig(output_dir / "v06_nearby_ebs.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ V06 nearby EBs plot saved")


def generate_v07_plot(output_dir: Path) -> None:
    """Generate V07 ExoFOP card plot."""
    from tess_vetter.plotting import plot_exofop_card

    result = ok_result(
        id="V07",
        name="ExoFOP TOI",
        metrics={"found": True},
        raw={
            "plot_data": {
                "version": 1,
                "tic_id": 123456789,
                "found": True,
                "toi": 1234.01,
                "tfopwg_disposition": "PC",
                "planet_disposition": "Candidate",
                "comments": "Strong transit signal, follow-up recommended",
            }
        },
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    plot_exofop_card(result, ax=ax)
    fig.savefig(output_dir / "v07_exofop_card.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ V07 ExoFOP card plot saved")


def generate_v10_plot(output_dir: Path) -> None:
    """Generate V10 aperture curve plot."""
    from tess_vetter.plotting import plot_aperture_curve

    result = ok_result(
        id="V10",
        name="Aperture Dependence",
        metrics={"depth_slope": 0.02},
        raw={
            "plot_data": {
                "version": 1,
                "aperture_radii_px": [1.0, 2.0, 3.0, 4.0, 5.0],
                "depths_ppm": [255.0, 250.0, 248.0, 245.0, 240.0],
                "depth_errs_ppm": [20.0, 15.0, 12.0, 10.0, 10.0],
            }
        },
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_aperture_curve(result, ax=ax)
    fig.savefig(output_dir / "v10_aperture_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ V10 aperture curve plot saved")


def generate_v11_plot(output_dir: Path) -> None:
    """Generate V11 ModShift plot."""
    from tess_vetter.plotting import plot_modshift

    np.random.seed(42)
    phase_bins = np.linspace(0, 1, 200)
    # Create ModShift-like signal with primary and secondary peaks
    periodogram = np.random.normal(0, 0.1, 200)
    periodogram[10:15] = 0.8  # Primary peak near phase 0.05
    periodogram[100:105] = 0.4  # Secondary peak near phase 0.5

    result = ok_result(
        id="V11",
        name="ModShift",
        metrics={"modshift_fa": 0.01},
        raw={
            "plot_data": {
                "version": 1,
                "phase_bins": phase_bins.tolist(),
                "periodogram": periodogram.tolist(),
                "primary_phase": 0.0625,
                "primary_signal": 0.8,
                "secondary_phase": 0.5125,
                "secondary_signal": 0.4,
            }
        },
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_modshift(result, ax=ax)
    fig.savefig(output_dir / "v11_modshift.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ V11 ModShift plot saved")


def generate_v12_plot(output_dir: Path) -> None:
    """Generate V12 SWEET plot."""
    from tess_vetter.plotting import plot_sweet

    np.random.seed(42)
    phase = np.linspace(0, 1, 300)
    flux = 1.0 + np.random.normal(0, 0.001, 300)
    # Add subtle sinusoidal variation at P
    flux += 0.0005 * np.sin(2 * np.pi * phase)

    # Create sinusoidal fit at P
    at_period_fit = 1.0 + 0.0005 * np.sin(2 * np.pi * phase)

    result = ok_result(
        id="V12",
        name="SWEET",
        metrics={"sweet_snr": 1.5},
        raw={
            "plot_data": {
                "version": 1,
                "phase": phase.tolist(),
                "flux": flux.tolist(),
                "half_period_fit": None,
                "at_period_fit": at_period_fit.tolist(),
                "double_period_fit": None,
                "snr_at_period": 1.5,
            }
        },
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_sweet(result, ax=ax)
    fig.savefig(output_dir / "v12_sweet.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ V12 SWEET plot saved")


def generate_v13_plot(output_dir: Path) -> None:
    """Generate V13 data gaps plot."""
    from tess_vetter.plotting import plot_data_gaps

    result = ok_result(
        id="V13",
        name="Data Gaps",
        metrics={"max_missing_frac": 0.15},
        raw={
            "plot_data": {
                "version": 1,
                "epoch_centers_btjd": [2459000.0 + i * 14.24 for i in range(8)],
                "coverage_fractions": [0.95, 0.88, 1.0, 0.85, 0.92, 0.78, 0.99, 0.90],
                "transit_window_hours": 6.0,
            }
        },
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_data_gaps(result, ax=ax)
    fig.savefig(output_dir / "v13_data_gaps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ V13 data gaps plot saved")


def generate_v15_plot(output_dir: Path) -> None:
    """Generate V15 asymmetry plot."""
    from tess_vetter.plotting import plot_asymmetry

    np.random.seed(42)
    phase = np.linspace(-0.1, 0.1, 200)
    flux = 1.0 + np.random.normal(0, 0.001, 200)
    # Add slight asymmetry
    flux[phase < -0.03] += 0.0002
    flux[phase > 0.03] -= 0.0001

    result = ok_result(
        id="V15",
        name="Asymmetry",
        metrics={"asymmetry_sigma": 1.8},
        raw={
            "plot_data": {
                "version": 1,
                "phase": phase.tolist(),
                "flux": flux.tolist(),
                "left_bin_mean": 1.0002,
                "right_bin_mean": 0.9999,
                "left_bin_phase_range": [-0.08, -0.04],
                "right_bin_phase_range": [0.04, 0.08],
            }
        },
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_asymmetry(result, ax=ax)
    fig.savefig(output_dir / "v15_asymmetry.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ V15 asymmetry plot saved")


def generate_v16_plot(output_dir: Path) -> None:
    """Generate V16 model comparison plot."""
    from tess_vetter.plotting import plot_model_comparison

    np.random.seed(42)
    phase = np.linspace(-0.1, 0.1, 200)
    flux = 1.0 + np.random.normal(0, 0.001, 200)
    # Add transit dip
    flux[np.abs(phase) < 0.02] -= 0.002

    # Create model arrays
    transit_model = np.ones(200)
    transit_model[np.abs(phase) < 0.02] = 0.998

    eb_model = np.ones(200)
    eb_model[np.abs(phase) < 0.015] = 0.997

    sinusoid_model = 1.0 - 0.001 * np.cos(2 * np.pi * phase / 0.1)

    result = ok_result(
        id="V16",
        name="Model Competition",
        metrics={"winner": "transit", "delta_bic": 15.2},
        raw={
            "plot_data": {
                "version": 1,
                "phase": phase.tolist(),
                "flux": flux.tolist(),
                "transit_model": transit_model.tolist(),
                "eb_like_model": eb_model.tolist(),
                "sinusoid_model": sinusoid_model.tolist(),
            }
        },
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_model_comparison(result, ax=ax)
    fig.savefig(output_dir / "v16_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ V16 model comparison plot saved")


def generate_v17_plot(output_dir: Path) -> None:
    """Generate V17 ephemeris reliability plot."""
    from tess_vetter.plotting import plot_ephemeris_reliability

    np.random.seed(42)
    phase_shifts = np.linspace(-0.5, 0.5, 100)
    # Create null scores with peak at 0
    null_scores = np.random.normal(0.5, 0.1, 100)
    null_scores[45:55] += 0.5  # Peak at center (on-ephemeris)

    result = ok_result(
        id="V17",
        name="Ephemeris Reliability",
        metrics={"phase_shift_null_p_value": 0.02},
        raw={
            "plot_data": {
                "version": 1,
                "phase_shifts": phase_shifts.tolist(),
                "null_scores": null_scores.tolist(),
                "period_neighborhood": [],
                "neighborhood_scores": [],
            }
        },
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_ephemeris_reliability(result, ax=ax)
    fig.savefig(output_dir / "v17_ephemeris_reliability.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ V17 ephemeris reliability plot saved")


def generate_v18_plot(output_dir: Path) -> None:
    """Generate V18 sensitivity sweep plot."""
    from tess_vetter.plotting import plot_sensitivity_sweep

    sweep_table = [
        {
            "variant_id": "ds=1|none|none",
            "status": "ok",
            "backend": "cpu",
            "runtime_seconds": 0.1,
            "n_points_used": 1000,
            "downsample_factor": 1,
            "outlier_policy": "none",
            "detrender": "none",
            "score": 0.95,
            "depth_hat_ppm": 2200.0,
            "depth_err_ppm": 120.0,
            "warnings": [],
            "failure_reason": None,
            "variant_config": {},
            "gp_hyperparams": None,
            "gp_fit_diagnostics": None,
        },
        {
            "variant_id": "ds=2|sigma_clip_4|running_median_0.5d",
            "status": "ok",
            "backend": "cpu",
            "runtime_seconds": 0.2,
            "n_points_used": 500,
            "downsample_factor": 2,
            "outlier_policy": "sigma_clip_4",
            "detrender": "running_median_0.5d",
            "score": 0.88,
            "depth_hat_ppm": 2100.0,
            "depth_err_ppm": 140.0,
            "warnings": [],
            "failure_reason": None,
            "variant_config": {},
            "gp_hyperparams": None,
            "gp_fit_diagnostics": None,
        },
        {
            "variant_id": "ds=5|none|none",
            "status": "failed",
            "backend": "cpu",
            "runtime_seconds": 0.05,
            "n_points_used": 200,
            "downsample_factor": 5,
            "outlier_policy": "none",
            "detrender": "none",
            "score": None,
            "depth_hat_ppm": None,
            "depth_err_ppm": None,
            "warnings": ["timeout"],
            "failure_reason": "timeout",
            "variant_config": {},
            "gp_hyperparams": None,
            "gp_fit_diagnostics": None,
        },
    ]

    result = ok_result(
        id="V18",
        name="Sensitivity Sweep",
        metrics={"n_variants_total": 3, "n_variants_ok": 2, "metric_variance": 0.02},
        raw={
            "plot_data": {
                "version": 1,
                "stable": True,
                "n_variants_total": 3,
                "n_variants_ok": 2,
                "sweep_table": sweep_table,
            }
        },
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_sensitivity_sweep(result, ax=ax)
    fig.savefig(output_dir / "v18_sensitivity_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ V18 sensitivity sweep plot saved")


def generate_v19_plot(output_dir: Path) -> None:
    """Generate V19 alias diagnostics plot."""
    from tess_vetter.plotting import plot_alias_diagnostics

    result = ok_result(
        id="V19",
        name="Alias Diagnostics",
        metrics={"best_harmonic": "P"},
        raw={
            "plot_data": {
                "version": 1,
                "harmonic_labels": ["P/3", "P/2", "P", "2P", "3P"],
                "harmonic_periods": [4.75, 7.12, 14.24, 28.48, 42.72],
                "harmonic_scores": [0.2, 0.35, 0.95, 0.4, 0.15],
            }
        },
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_alias_diagnostics(result, ax=ax)
    fig.savefig(output_dir / "v19_alias_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ V19 alias diagnostics plot saved")


def generate_v20_plot(output_dir: Path) -> None:
    """Generate V20 ghost features plot."""
    from tess_vetter.plotting import plot_ghost_features

    np.random.seed(42)
    diff_image = np.random.normal(0, 10, (11, 11))
    # Add signal at target
    diff_image[4:7, 4:7] += 50
    aperture_mask = np.zeros((11, 11), dtype=bool)
    aperture_mask[3:8, 3:8] = True

    result = ok_result(
        id="V20",
        name="Ghost Features",
        metrics={"ghost_score": 0.1},
        raw={
            "plot_data": {
                "version": 1,
                "difference_image": diff_image.tolist(),
                "aperture_mask": aperture_mask.tolist(),
                "in_aperture_depth": 0.002,
                "out_aperture_depth": 0.0001,
            }
        },
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    ax_out, _ = plot_ghost_features(result, ax=ax)
    fig.savefig(output_dir / "v20_ghost_features.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ V20 ghost features plot saved")


def generate_full_lightcurve_plot(output_dir: Path) -> None:
    """Generate full light curve plot."""
    from tess_vetter.plotting import plot_full_lightcurve
    from tess_vetter.api.types import LightCurve, Candidate, Ephemeris

    np.random.seed(42)
    # Create mock light curve with transits
    time = np.linspace(2459000, 2459100, 5000)
    period = 14.24
    t0 = 2459010.0

    flux = 1.0 + np.random.normal(0, 0.001, 5000)
    # Add transits
    for epoch in range(-1, 8):
        t_mid = t0 + epoch * period
        in_transit = np.abs(time - t_mid) < 0.1
        flux[in_transit] -= 0.002

    lc = LightCurve(time=time, flux=flux, flux_err=np.ones(5000) * 0.001)
    candidate = Candidate(
        ephemeris=Ephemeris(period_days=period, t0_btjd=t0, duration_hours=4.0),
        depth_ppm=2000,
    )

    fig, ax = plt.subplots(figsize=(12, 4))
    plot_full_lightcurve(lc, candidate=candidate, ax=ax)
    fig.savefig(output_dir / "full_lightcurve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ Full light curve plot saved")


def generate_phase_folded_plot(output_dir: Path) -> None:
    """Generate phase-folded transit plot."""
    from tess_vetter.plotting import plot_phase_folded
    from tess_vetter.api.types import LightCurve, Candidate, Ephemeris

    np.random.seed(42)
    # Create mock light curve with transits
    time = np.linspace(0, 100, 5000)
    period = 14.24
    t0 = 10.0

    flux = 1.0 + np.random.normal(0, 0.001, 5000)
    # Add transits
    phase = ((time - t0) / period) % 1
    phase[phase > 0.5] -= 1
    in_transit = np.abs(phase) < 0.01
    flux[in_transit] -= 0.002

    lc = LightCurve(time=time, flux=flux, flux_err=np.ones(5000) * 0.001)
    candidate = Candidate(
        ephemeris=Ephemeris(period_days=period, t0_btjd=t0, duration_hours=4.0),
        depth_ppm=2000,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_phase_folded(lc, candidate, ax=ax)
    fig.savefig(output_dir / "phase_folded.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ Phase-folded plot saved")


def generate_transit_fit_plot(output_dir: Path) -> None:
    """Generate transit fit overlay plot."""
    from tess_vetter.plotting import plot_transit_fit
    from tess_vetter.api.transit_fit import TransitFitResult

    # Synthetic fitted model + data in phase space
    phase = np.linspace(-0.1, 0.1, 400)
    flux_model = np.ones_like(phase)
    flux_model[np.abs(phase) < 0.012] = 0.998

    rng = np.random.default_rng(42)
    flux_data = flux_model + rng.normal(0, 0.0007, size=phase.size)

    fit_result = TransitFitResult(
        fit_method="optimize",
        rp_rs=0.045,
        rp_rs_err=0.003,
        a_rs=15.0,
        a_rs_err=1.0,
        inclination_deg=89.0,
        inclination_err=0.2,
        t0_offset=0.0,
        t0_offset_err=0.001,
        u1=0.3,
        u2=0.2,
        transit_depth_ppm=2000.0,
        duration_hours=4.0,
        impact_parameter=0.2,
        stellar_density_gcc=1.0,
        chi_squared=1.1,
        bic=1234.5,
        converged=True,
        phase=phase.tolist(),
        flux_model=flux_model.tolist(),
        flux_data=flux_data.tolist(),
        status="success",
        error_message=None,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_transit_fit(fit_result, ax=ax)
    fig.savefig(output_dir / "transit_fit.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ Transit fit plot saved")


def generate_dvr_summary(output_dir: Path) -> None:
    """Generate DVR summary plot."""
    from tess_vetter.plotting import plot_vetting_summary
    from tess_vetter.api.types import LightCurve, Candidate, Ephemeris
    from tess_vetter.validation.result_schema import VettingBundleResult

    np.random.seed(42)

    # Create mock light curve
    time = np.linspace(0, 100, 5000)
    period = 14.24
    t0 = 10.0
    flux = 1.0 + np.random.normal(0, 0.001, 5000)
    phase = ((time - t0) / period) % 1
    phase[phase > 0.5] -= 1
    in_transit = np.abs(phase) < 0.01
    flux[in_transit] -= 0.002

    lc = LightCurve(time=time, flux=flux, flux_err=np.ones(5000) * 0.001)
    candidate = Candidate(
        ephemeris=Ephemeris(period_days=period, t0_btjd=t0, duration_hours=4.0),
        depth_ppm=2000,
    )

    # Create mock bundle with minimal results
    v01 = ok_result("V01", "Odd-Even Depth", {"sigma_diff": 0.8}, raw={
        "plot_data": {
            "version": 1,
            "odd_epochs": [1, 3, 5],
            "odd_depths_ppm": [250.0, 252.0, 248.0],
            "odd_errs_ppm": [15.0, 14.0, 16.0],
            "even_epochs": [2, 4, 6],
            "even_depths_ppm": [249.0, 251.0, 250.0],
            "even_errs_ppm": [14.0, 15.0, 14.0],
            "mean_odd_ppm": 250.0,
            "mean_even_ppm": 250.0,
        }
    })

    bundle = VettingBundleResult(results=[v01])

    fig = plot_vetting_summary(bundle, lc, candidate, include_panels=["A", "B", "D", "H"])
    fig.savefig(output_dir / "dvr_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("✓ DVR summary plot saved")

def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate verification plots for plotting module.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help=(
            "Output directory for generated PNGs. "
            "Default: working_docs/image_support/verification/verification_plots/"
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    """Generate all verification plots."""
    args = parse_args(sys.argv[1:])
    print("Generating verification plots...")
    print("=" * 50)

    output_dir = create_output_dir(args.out_dir)
    print(f"Output directory: {output_dir}\n")

    # V01-V05: Light curve checks
    generate_v01_plot(output_dir)
    generate_v02_plot(output_dir)
    generate_v03_plot(output_dir)
    generate_v04_plot(output_dir)
    generate_v05_plot(output_dir)

    # V06-V07: Catalog checks
    generate_v06_plot(output_dir)
    generate_v07_plot(output_dir)

    # V08-V10: Pixel checks
    generate_v08_plot(output_dir)
    generate_v09_plot(output_dir)
    generate_v10_plot(output_dir)

    # V11-V12: Exovetter checks
    generate_v11_plot(output_dir)
    generate_v12_plot(output_dir)

    # V13, V15: False alarm checks
    generate_v13_plot(output_dir)
    generate_v15_plot(output_dir)

    # V16-V21: Extended checks
    generate_v16_plot(output_dir)
    generate_v17_plot(output_dir)
    generate_v18_plot(output_dir)
    generate_v19_plot(output_dir)
    generate_v20_plot(output_dir)
    generate_v21_plot(output_dir)

    # Transit and light curve visualizations
    generate_phase_folded_plot(output_dir)
    generate_transit_fit_plot(output_dir)
    generate_full_lightcurve_plot(output_dir)

    # DVR summary
    generate_dvr_summary(output_dir)

    print("\n" + "=" * 50)
    print(f"All plots saved to: {output_dir}")
    print("\nPlots generated:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
