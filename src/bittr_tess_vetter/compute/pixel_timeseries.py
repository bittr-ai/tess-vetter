"""Pixel time-series inference for transit host identification.

This module implements windowed pixel time-series modeling for transit host
hypothesis testing. It resolves blends that are ambiguous under difference
imaging by using the full temporal information in transit windows.

Phase 3.3 Transit-on-pixels Time-series Inference:
- Extract transit windows from TPF data
- Fit transit amplitude per hypothesis using WLS
- Aggregate evidence across windows for hypothesis comparison

CRITICAL: This module must remain pure compute:
- NO file I/O (open, Path, etc.)
- NO network access
- NO imports of astropy, lightkurve at module level
- ONLY numpy and scipy dependencies
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .prf_psf import PRFModel

__all__ = [
    # Data structures
    "TransitWindow",
    "PixelTimeseriesFit",
    "TimeseriesEvidence",
    "TimeseriesDiagnostics",
    # Window extraction
    "extract_transit_windows",
    # Fitting functions
    "fit_transit_amplitude_wls",
    "fit_all_hypotheses_timeseries",
    # Aggregation functions
    "aggregate_timeseries_evidence",
    "select_best_hypothesis_timeseries",
    # Diagnostics
    "compute_timeseries_diagnostics",
]


# =============================================================================
# Constants
# =============================================================================

DEFAULT_WINDOW_MARGIN: float = 2.0
"""Default margin multiplier for transit window extraction.

Window extends duration * margin on each side of transit center.
"""

DEFAULT_MIN_IN_TRANSIT: int = 3
"""Minimum in-transit points required per window."""

DEFAULT_BASELINE_ORDER: int = 0
"""Default baseline order (0 = constant, 1 = linear)."""

DEFAULT_MARGIN_THRESHOLD: float = 2.0
"""Default chi-squared difference threshold for hypothesis selection."""


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class TransitWindow:
    """Data for a single transit window.

    Contains the pixel time-series data around a single transit event,
    along with masks and metadata for fitting.

    Attributes
    ----------
    transit_idx : int
        Zero-based index of this transit event.
    time : NDArray[np.float64]
        Time array in BTJD, shape (n_cadences,).
    pixels : NDArray[np.float64]
        Pixel flux cube, shape (n_cadences, n_rows, n_cols).
    errors : NDArray[np.float64]
        Flux uncertainties, shape (n_cadences, n_rows, n_cols) or scalar.
    in_transit_mask : NDArray[np.bool_]
        Boolean mask for in-transit cadences, shape (n_cadences,).
    t0_expected : float
        Expected transit center time in BTJD.
    """

    transit_idx: int
    time: NDArray[np.float64]
    pixels: NDArray[np.float64]
    errors: NDArray[np.float64]
    in_transit_mask: NDArray[np.bool_]
    t0_expected: float


@dataclass
class PixelTimeseriesFit:
    """Result of pixel time-series fit for one hypothesis.

    Contains the fitted transit amplitude and fit quality metrics
    for a single hypothesis on a single transit window.

    Attributes
    ----------
    source_id : str
        Source identifier for the hypothesis.
    amplitude : float
        Fitted transit amplitude (flux drop, typically negative).
    amplitude_err : float
        Uncertainty on the fitted amplitude.
    chi2 : float
        Chi-squared statistic for the fit.
    dof : int
        Degrees of freedom.
    residual_rms : float
        RMS of residuals after fit.
    per_pixel_residuals : NDArray[np.float64] | None
        Optional per-pixel residual summary (mean over time), shape (n_rows, n_cols).
    """

    source_id: str
    amplitude: float
    amplitude_err: float
    chi2: float
    dof: int
    residual_rms: float
    per_pixel_residuals: NDArray[np.float64] | None = None


@dataclass
class TimeseriesEvidence:
    """Aggregated time-series evidence for hypothesis comparison.

    Combines results from multiple transit windows into overall
    evidence for a hypothesis.

    Attributes
    ----------
    source_id : str
        Source identifier for the hypothesis.
    total_chi2 : float
        Sum of chi-squared across all windows.
    total_dof : int
        Sum of degrees of freedom across all windows.
    mean_amplitude : float
        Mean fitted amplitude across windows.
    amplitude_scatter : float
        Standard deviation of per-window amplitudes.
    n_windows_fitted : int
        Number of successfully fitted windows.
    log_likelihood : float
        Approximate log-likelihood for joint inference.
    """

    source_id: str
    total_chi2: float
    total_dof: int
    mean_amplitude: float
    amplitude_scatter: float
    n_windows_fitted: int
    log_likelihood: float


@dataclass
class TimeseriesDiagnostics:
    """Debug artifacts for time-series inference.

    Provides detailed per-window metrics for debugging and
    diagnosing fit quality issues.

    Attributes
    ----------
    per_window_amplitudes : dict[str, list[float]]
        Mapping of source_id to list of fitted amplitudes per window.
    per_window_chi2 : dict[str, list[float]]
        Mapping of source_id to list of chi-squared values per window.
    outlier_windows : list[int]
        Indices of windows with anomalous residuals.
    warnings : list[str]
        Diagnostic warnings encountered.
    """

    per_window_amplitudes: dict[str, list[float]] = field(default_factory=dict)
    per_window_chi2: dict[str, list[float]] = field(default_factory=dict)
    outlier_windows: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# =============================================================================
# Window Extraction
# =============================================================================


def extract_transit_windows(
    tpf_data: NDArray[np.float64],
    time: NDArray[np.float64],
    period: float,
    t0: float,
    duration_hours: float,
    *,
    errors: NDArray[np.float64] | float | None = None,
    window_margin: float = DEFAULT_WINDOW_MARGIN,
    min_in_transit: int = DEFAULT_MIN_IN_TRANSIT,
) -> list[TransitWindow]:
    """Extract transit windows from TPF data.

    Identifies transit events in the time series and extracts
    pixel data around each transit for fitting.

    Parameters
    ----------
    tpf_data : NDArray[np.float64]
        Pixel flux cube, shape (n_cadences, n_rows, n_cols).
    time : NDArray[np.float64]
        Time array in BTJD, shape (n_cadences,).
    period : float
        Orbital period in days.
    t0 : float
        Transit epoch in BTJD.
    duration_hours : float
        Full transit duration in hours.
    errors : NDArray[np.float64] | float | None, optional
        Flux uncertainties. Can be:
        - 3D array matching tpf_data shape
        - Scalar value applied to all pixels
        - None (defaults to sqrt(flux) Poisson noise estimate)
    window_margin : float, optional
        Window extends duration * margin on each side.
        Default is DEFAULT_WINDOW_MARGIN (2.0).
    min_in_transit : int, optional
        Minimum in-transit points required per window.
        Default is DEFAULT_MIN_IN_TRANSIT (3).

    Returns
    -------
    list[TransitWindow]
        List of extracted transit windows, sorted by transit index.
        Empty list if no valid windows found.

    Notes
    -----
    Windows are centered on expected transit mid-times computed from
    the ephemeris (t0, period). Each window extends duration * margin
    on each side of the transit center.

    Windows with fewer than min_in_transit in-transit points are skipped.
    """
    if tpf_data.ndim != 3:
        raise ValueError(f"tpf_data must be 3D, got shape {tpf_data.shape}")

    if len(time) != tpf_data.shape[0]:
        raise ValueError(
            f"time length ({len(time)}) must match tpf_data first dimension ({tpf_data.shape[0]})"
        )

    duration_days = duration_hours / 24.0
    window_half_width = duration_days * window_margin

    # Find transit mid-times within the observation window
    t_min, t_max = float(np.min(time)), float(np.max(time))

    # Compute first transit after t_min
    n_start = int(np.ceil((t_min - t0) / period))
    n_end = int(np.floor((t_max - t0) / period))

    # Handle error array
    if errors is None:
        # Estimate Poisson noise from flux (clip to positive)
        errors_arr = np.sqrt(np.clip(tpf_data, 1.0, None))
    elif isinstance(errors, (int, float)):
        errors_arr = np.full_like(tpf_data, errors)
    else:
        errors_arr = np.asarray(errors, dtype=np.float64)
        if errors_arr.shape != tpf_data.shape:
            raise ValueError(
                f"errors shape {errors_arr.shape} must match tpf_data shape {tpf_data.shape}"
            )

    windows: list[TransitWindow] = []

    for n in range(n_start, n_end + 1):
        t_transit = t0 + n * period

        # Window bounds
        t_window_start = t_transit - window_half_width
        t_window_end = t_transit + window_half_width

        # Find cadences in window
        window_mask = (time >= t_window_start) & (time <= t_window_end)
        n_cadences = int(np.sum(window_mask))

        if n_cadences < min_in_transit:
            continue

        # Extract window data
        time_window = time[window_mask]
        pixels_window = tpf_data[window_mask]
        errors_window = errors_arr[window_mask]

        # Compute in-transit mask
        half_duration = duration_days / 2.0
        in_transit_mask = np.abs(time_window - t_transit) <= half_duration

        n_in_transit = int(np.sum(in_transit_mask))
        if n_in_transit < min_in_transit:
            continue

        windows.append(
            TransitWindow(
                transit_idx=n - n_start,
                time=time_window.astype(np.float64),
                pixels=pixels_window.astype(np.float64),
                errors=errors_window.astype(np.float64),
                in_transit_mask=in_transit_mask,
                t0_expected=t_transit,
            )
        )

    return windows


# =============================================================================
# WLS Fitting
# =============================================================================


def fit_transit_amplitude_wls(
    window: TransitWindow,
    hypothesis_row: float,
    hypothesis_col: float,
    prf_model: PRFModel,
    *,
    fit_baseline: bool = True,
    baseline_order: int = DEFAULT_BASELINE_ORDER,
) -> PixelTimeseriesFit:
    """Fit transit amplitude using weighted least squares.

    Fits a linear model to the pixel time-series where the transit
    signal is modeled as a flux drop weighted by the PRF at the
    hypothesis source position.

    Model per pixel: flux = baseline + amplitude * prf_weight * in_transit

    Parameters
    ----------
    window : TransitWindow
        Transit window data to fit.
    hypothesis_row : float
        Row coordinate of the hypothesis source position.
    hypothesis_col : float
        Column coordinate of the hypothesis source position.
    prf_model : PRFModel
        PRF model for computing pixel weights.
    fit_baseline : bool, optional
        Whether to fit per-pixel baseline. Default True.
    baseline_order : int, optional
        Baseline polynomial order (0 = constant, 1 = linear).
        Default is DEFAULT_BASELINE_ORDER (0).

    Returns
    -------
    PixelTimeseriesFit
        Fitted amplitude and quality metrics.

    Notes
    -----
    The WLS fit solves for the transit amplitude that best explains
    the in-transit flux deficit across all pixels, weighted by the
    PRF contribution from the hypothesis source.

    For baseline_order=0, each pixel has a constant baseline.
    For baseline_order=1, each pixel has a linear time trend.

    The amplitude is fitted globally across all pixels (single parameter).
    """
    pixels = window.pixels  # (n_cadences, n_rows, n_cols)
    errors = window.errors
    in_transit = window.in_transit_mask  # (n_cadences,)
    time = window.time

    n_cadences, n_rows, n_cols = pixels.shape
    n_pixels = n_rows * n_cols

    # Get PRF weights for hypothesis position
    prf_weights = prf_model.evaluate(
        hypothesis_row,
        hypothesis_col,
        (n_rows, n_cols),
        normalize=True,
    )  # shape (n_rows, n_cols)

    # Flatten spatial dimensions
    pixels_flat = pixels.reshape(n_cadences, n_pixels)  # (n_cadences, n_pixels)
    errors_flat = errors.reshape(n_cadences, n_pixels)
    prf_weights_flat = prf_weights.flatten()  # (n_pixels,)

    # Build design matrix
    # We fit: flux[t, p] = baseline[p] + amplitude * prf_weights[p] * in_transit[t]
    # For baseline_order=0: baseline[p] is a constant per pixel
    # For baseline_order=1: baseline[p] = b0[p] + b1[p] * (t - t_mean)

    # Number of parameters:
    # - amplitude: 1
    # - baseline per pixel: n_pixels * (baseline_order + 1)

    t_mean = float(np.mean(time))
    t_normalized = (time - t_mean) / max(1e-10, float(np.std(time)))

    if fit_baseline:
        if baseline_order == 0:
            # Design matrix columns: [baseline_1, ..., baseline_N, amplitude]
            # For each cadence t and pixel p:
            # flux[t,p] = baseline[p] + amplitude * prf[p] * in_transit[t]

            n_params = n_pixels + 1

            # Build design matrix: (n_cadences * n_pixels, n_params)
            design_mat = np.zeros((n_cadences * n_pixels, n_params), dtype=np.float64)

            # Baseline columns (one per pixel)
            for p in range(n_pixels):
                design_mat[p::n_pixels, p] = 1.0

            # Amplitude column
            for t in range(n_cadences):
                if in_transit[t]:
                    design_mat[t * n_pixels : (t + 1) * n_pixels, -1] = prf_weights_flat

        else:
            # baseline_order == 1
            # baseline[p] = b0[p] + b1[p] * t_normalized
            n_params = 2 * n_pixels + 1

            design_mat = np.zeros((n_cadences * n_pixels, n_params), dtype=np.float64)

            # b0 columns
            for p in range(n_pixels):
                design_mat[p::n_pixels, p] = 1.0

            # b1 columns
            for t in range(n_cadences):
                for p in range(n_pixels):
                    design_mat[t * n_pixels + p, n_pixels + p] = t_normalized[t]

            # Amplitude column
            for t in range(n_cadences):
                if in_transit[t]:
                    design_mat[t * n_pixels : (t + 1) * n_pixels, -1] = prf_weights_flat
    else:
        # No baseline fitting - just amplitude
        n_params = 1
        design_mat = np.zeros((n_cadences * n_pixels, 1), dtype=np.float64)

        for t in range(n_cadences):
            if in_transit[t]:
                design_mat[t * n_pixels : (t + 1) * n_pixels, 0] = prf_weights_flat

    # Flatten observations
    y = pixels_flat.flatten()  # (n_cadences * n_pixels,)
    w = 1.0 / (errors_flat.flatten() ** 2 + 1e-30)  # weights

    # Mask out bad pixels (NaN, inf)
    valid_mask = np.isfinite(y) & np.isfinite(w) & (w > 0)
    if np.sum(valid_mask) < n_params:
        # Not enough valid points
        return PixelTimeseriesFit(
            source_id="",
            amplitude=0.0,
            amplitude_err=float("inf"),
            chi2=float("inf"),
            dof=0,
            residual_rms=float("inf"),
            per_pixel_residuals=None,
        )

    design_valid = design_mat[valid_mask]
    y_valid = y[valid_mask]
    w_valid = w[valid_mask]

    # Weighted least squares: (X^T W X) beta = X^T W y
    xt_w_x = design_valid.T @ (design_valid * w_valid[:, np.newaxis])
    xt_w_y = design_valid.T @ (y_valid * w_valid)

    try:
        # Solve with regularization for numerical stability
        reg = 1e-10 * np.eye(n_params)
        beta = np.linalg.solve(xt_w_x + reg, xt_w_y)
    except np.linalg.LinAlgError:
        return PixelTimeseriesFit(
            source_id="",
            amplitude=0.0,
            amplitude_err=float("inf"),
            chi2=float("inf"),
            dof=0,
            residual_rms=float("inf"),
            per_pixel_residuals=None,
        )

    # Extract amplitude (last parameter)
    amplitude = float(beta[-1])

    # Compute residuals
    y_pred = design_valid @ beta
    residuals = y_valid - y_pred
    weighted_residuals = residuals * np.sqrt(w_valid)

    chi2 = float(np.sum(weighted_residuals**2))
    dof = int(np.sum(valid_mask)) - n_params

    # Estimate amplitude uncertainty from covariance
    try:
        cov = np.linalg.inv(xt_w_x + reg)
        amplitude_err = float(np.sqrt(cov[-1, -1]))
    except np.linalg.LinAlgError:
        amplitude_err = float("inf")

    residual_rms = float(np.std(residuals))

    # Compute per-pixel residual summary
    residuals_full = np.full(n_cadences * n_pixels, np.nan)
    residuals_full[valid_mask] = residuals
    residuals_reshaped = residuals_full.reshape(n_cadences, n_rows, n_cols)
    per_pixel_residuals = np.nanmean(residuals_reshaped, axis=0)

    return PixelTimeseriesFit(
        source_id="",  # Will be set by caller
        amplitude=amplitude,
        amplitude_err=amplitude_err,
        chi2=chi2,
        dof=max(1, dof),
        residual_rms=residual_rms,
        per_pixel_residuals=per_pixel_residuals,
    )


def fit_all_hypotheses_timeseries(
    windows: list[TransitWindow],
    hypotheses: list[dict[str, float | str]],
    prf_model: PRFModel,
    *,
    fit_baseline: bool = True,
    baseline_order: int = DEFAULT_BASELINE_ORDER,
) -> dict[str, list[PixelTimeseriesFit]]:
    """Fit all hypotheses across all transit windows.

    Parameters
    ----------
    windows : list[TransitWindow]
        List of transit windows to fit.
    hypotheses : list[dict[str, float | str]]
        List of hypothesis dictionaries with keys:
        - source_id: str
        - row: float (pixel row coordinate)
        - col: float (pixel column coordinate)
    prf_model : PRFModel
        PRF model for computing pixel weights.
    fit_baseline : bool, optional
        Whether to fit per-pixel baseline. Default True.
    baseline_order : int, optional
        Baseline polynomial order. Default 0.

    Returns
    -------
    dict[str, list[PixelTimeseriesFit]]
        Mapping of source_id to list of per-window fits.
    """
    results: dict[str, list[PixelTimeseriesFit]] = {}

    for hyp in hypotheses:
        source_id = str(hyp.get("source_id", "unknown"))
        row = float(hyp.get("row", 0.0))
        col = float(hyp.get("col", 0.0))

        fits: list[PixelTimeseriesFit] = []

        for window in windows:
            fit = fit_transit_amplitude_wls(
                window,
                row,
                col,
                prf_model,
                fit_baseline=fit_baseline,
                baseline_order=baseline_order,
            )
            # Set source_id
            fit = PixelTimeseriesFit(
                source_id=source_id,
                amplitude=fit.amplitude,
                amplitude_err=fit.amplitude_err,
                chi2=fit.chi2,
                dof=fit.dof,
                residual_rms=fit.residual_rms,
                per_pixel_residuals=fit.per_pixel_residuals,
            )
            fits.append(fit)

        results[source_id] = fits

    return results


# =============================================================================
# Evidence Aggregation
# =============================================================================


def aggregate_timeseries_evidence(
    fits: list[PixelTimeseriesFit],
) -> TimeseriesEvidence:
    """Aggregate per-window fits into overall evidence.

    Combines fit results from multiple transit windows into
    summary statistics for hypothesis comparison.

    Parameters
    ----------
    fits : list[PixelTimeseriesFit]
        List of per-window fits for a single hypothesis.

    Returns
    -------
    TimeseriesEvidence
        Aggregated evidence for the hypothesis.
    """
    if not fits:
        return TimeseriesEvidence(
            source_id="",
            total_chi2=float("inf"),
            total_dof=0,
            mean_amplitude=0.0,
            amplitude_scatter=float("inf"),
            n_windows_fitted=0,
            log_likelihood=float("-inf"),
        )

    source_id = fits[0].source_id

    # Filter valid fits
    valid_fits = [f for f in fits if np.isfinite(f.chi2) and f.dof > 0]
    n_windows = len(valid_fits)

    if n_windows == 0:
        return TimeseriesEvidence(
            source_id=source_id,
            total_chi2=float("inf"),
            total_dof=0,
            mean_amplitude=0.0,
            amplitude_scatter=float("inf"),
            n_windows_fitted=0,
            log_likelihood=float("-inf"),
        )

    total_chi2 = sum(f.chi2 for f in valid_fits)
    total_dof = sum(f.dof for f in valid_fits)

    amplitudes = [f.amplitude for f in valid_fits]
    mean_amplitude = float(np.mean(amplitudes))
    amplitude_scatter = float(np.std(amplitudes)) if len(amplitudes) > 1 else 0.0

    # Approximate log-likelihood: -0.5 * chi2
    log_likelihood = -0.5 * total_chi2

    return TimeseriesEvidence(
        source_id=source_id,
        total_chi2=total_chi2,
        total_dof=total_dof,
        mean_amplitude=mean_amplitude,
        amplitude_scatter=amplitude_scatter,
        n_windows_fitted=n_windows,
        log_likelihood=log_likelihood,
    )


def select_best_hypothesis_timeseries(
    evidence: dict[str, TimeseriesEvidence],
    *,
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
) -> tuple[str, str, float]:
    """Select best hypothesis from time-series evidence.

    Compares hypotheses using their aggregated chi-squared statistics
    and determines the verdict based on the margin.

    Parameters
    ----------
    evidence : dict[str, TimeseriesEvidence]
        Mapping of source_id to TimeseriesEvidence.
    margin_threshold : float, optional
        Delta chi-squared threshold for resolved verdict.
        Default is DEFAULT_MARGIN_THRESHOLD (2.0).

    Returns
    -------
    tuple[str, str, float]
        (best_source_id, verdict, delta_chi2)
        - best_source_id: Source ID with lowest chi-squared
        - verdict: "ON_TARGET" | "OFF_TARGET" | "AMBIGUOUS"
        - delta_chi2: Chi-squared difference to runner-up
    """
    if not evidence:
        return ("", "AMBIGUOUS", 0.0)

    # Sort by total_chi2 (lower is better)
    sorted_evidence = sorted(
        evidence.items(),
        key=lambda x: (x[1].total_chi2, x[0]),  # tie-break by source_id
    )

    best_source_id = sorted_evidence[0][0]
    best_chi2 = sorted_evidence[0][1].total_chi2

    if len(sorted_evidence) > 1:
        runner_up_chi2 = sorted_evidence[1][1].total_chi2
        delta_chi2 = runner_up_chi2 - best_chi2
    else:
        delta_chi2 = float("inf")

    # Determine verdict
    is_target = "target" in best_source_id.lower()

    if not np.isfinite(delta_chi2) or delta_chi2 >= margin_threshold:
        verdict = "ON_TARGET" if is_target else "OFF_TARGET"
    else:
        verdict = "AMBIGUOUS"

    return (best_source_id, verdict, float(delta_chi2) if np.isfinite(delta_chi2) else 0.0)


# =============================================================================
# Diagnostics
# =============================================================================


def compute_timeseries_diagnostics(
    fits: dict[str, list[PixelTimeseriesFit]],
    windows: list[TransitWindow],
    *,
    outlier_threshold: float = 3.0,
) -> TimeseriesDiagnostics:
    """Compute diagnostic artifacts for time-series inference.

    Analyzes fit results to identify problematic windows and
    generate debugging information.

    Parameters
    ----------
    fits : dict[str, list[PixelTimeseriesFit]]
        Mapping of source_id to list of per-window fits.
    windows : list[TransitWindow]
        Original transit windows (for cross-reference).
    outlier_threshold : float, optional
        Number of sigma for outlier detection. Default 3.0.

    Returns
    -------
    TimeseriesDiagnostics
        Diagnostic artifacts including per-window metrics and warnings.
    """
    per_window_amplitudes: dict[str, list[float]] = {}
    per_window_chi2: dict[str, list[float]] = {}
    warnings: list[str] = []

    for source_id, source_fits in fits.items():
        per_window_amplitudes[source_id] = [f.amplitude for f in source_fits]
        per_window_chi2[source_id] = [f.chi2 for f in source_fits]

    # Identify outlier windows based on chi-squared
    # Use the best hypothesis (lowest mean chi2) as reference
    outlier_windows: list[int] = []

    if fits:
        # Find hypothesis with lowest mean chi2
        mean_chi2_by_hyp = {
            sid: float(np.mean([c for c in chi2_list if np.isfinite(c)]))
            for sid, chi2_list in per_window_chi2.items()
            if any(np.isfinite(c) for c in chi2_list)
        }

        if mean_chi2_by_hyp:
            best_hyp = min(mean_chi2_by_hyp.keys(), key=lambda k: mean_chi2_by_hyp[k])
            best_chi2_list = per_window_chi2[best_hyp]

            # Compute median and MAD for outlier detection
            valid_chi2 = [c for c in best_chi2_list if np.isfinite(c)]
            if len(valid_chi2) > 2:
                median_chi2 = float(np.median(valid_chi2))
                mad_chi2 = float(np.median(np.abs(np.array(valid_chi2) - median_chi2)))

                if mad_chi2 > 0:
                    for i, chi2_val in enumerate(best_chi2_list):
                        if np.isfinite(chi2_val):
                            z_score = abs(chi2_val - median_chi2) / (1.4826 * mad_chi2)
                            if z_score > outlier_threshold:
                                outlier_windows.append(i)

    # Generate warnings
    if outlier_windows:
        warnings.append(f"Found {len(outlier_windows)} outlier window(s): {outlier_windows}")

    for source_id, amplitudes in per_window_amplitudes.items():
        valid_amps = [a for a in amplitudes if np.isfinite(a)]
        if len(valid_amps) > 1:
            scatter = float(np.std(valid_amps))
            mean_amp = float(np.mean(valid_amps))
            if abs(mean_amp) > 0 and scatter / abs(mean_amp) > 0.5:
                warnings.append(
                    f"High amplitude scatter for {source_id}: scatter/mean = {scatter / abs(mean_amp):.2f}"
                )

    return TimeseriesDiagnostics(
        per_window_amplitudes=per_window_amplitudes,
        per_window_chi2=per_window_chi2,
        outlier_windows=outlier_windows,
        warnings=warnings,
    )
