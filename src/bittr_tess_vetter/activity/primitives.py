"""Activity primitives for stellar characterization.

Pure-compute functions for:
- Flare detection using sigma-clipping above local baseline
- Rotation period measurement with uncertainty estimation
- Variability classification (spotted_rotator, pulsator, etc.)
- Photometric activity index computation
- Flare masking/interpolation

All functions use pure numpy/scipy with no I/O operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import signal

from bittr_tess_vetter.activity.result import Flare

if TYPE_CHECKING:
    from numpy.typing import NDArray


def detect_flares(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    flux_err: NDArray[np.float64],
    sigma_threshold: float = 5.0,
    min_duration_minutes: float = 1.0,
    baseline_window_hours: float = 6.0,
) -> list[Flare]:
    """Detect stellar flares using sigma-clipping above local baseline.

    Flares are identified as positive excursions above a rolling median
    baseline that exceed sigma_threshold standard deviations.

    Args:
        time: Time array, in days (e.g., BTJD)
        flux: Normalized flux array (median ~1.0)
        flux_err: Flux uncertainties
        sigma_threshold: Detection threshold in sigma (default 5.0)
        min_duration_minutes: Minimum flare duration to consider, in minutes (default 1.0)
        baseline_window_hours: Window for computing local baseline, in hours (default 6.0)

    Returns:
        List of detected Flare objects, sorted by peak time.
    """
    if len(time) < 100:
        return []

    # Estimate cadence from median time differences
    time_diffs = np.diff(time)
    cadence_days = float(np.median(time_diffs[time_diffs < 0.5]))  # Ignore gaps
    cadence_minutes = cadence_days * 24.0 * 60.0

    # Convert window to number of points (odd for symmetric median filter)
    window_points = int(baseline_window_hours * 60.0 / cadence_minutes)
    window_points = max(11, window_points | 1)  # Ensure odd and at least 11

    # Compute rolling median baseline (handle edges with mode='nearest')
    baseline = _rolling_median(flux, window_points)

    # Compute residuals and local scatter
    residuals = flux - baseline
    local_scatter = _rolling_mad(residuals, window_points)

    # Avoid division by zero
    local_scatter = np.maximum(local_scatter, 1e-6)

    # Find points above threshold
    sigma_values = residuals / local_scatter
    above_threshold = sigma_values > sigma_threshold

    # Group consecutive detections into flare events
    flares = _group_flare_events(
        time,
        flux,
        flux_err,
        above_threshold,
        baseline,
        sigma_values,
        min_duration_minutes,
        cadence_minutes,
    )

    return flares


def _rolling_median(data: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    """Compute rolling median with edge handling.

    Args:
        data: Input array
        window: Window size (should be odd)

    Returns:
        Rolling median array same length as input
    """
    # Pad the array to handle edges
    half_window = window // 2
    padded = np.pad(data, (half_window, half_window), mode="reflect")

    # Compute rolling median using a simple loop (efficient for typical window sizes)
    result = np.zeros_like(data)
    for i in range(len(data)):
        result[i] = np.median(padded[i : i + window])

    return result


def _rolling_mad(data: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    """Compute rolling median absolute deviation (MAD).

    Args:
        data: Input array
        window: Window size (should be odd)

    Returns:
        Rolling MAD array scaled to sigma-equivalent (x1.4826)
    """
    half_window = window // 2
    padded = np.pad(data, (half_window, half_window), mode="reflect")

    result = np.zeros_like(data)
    for i in range(len(data)):
        chunk = padded[i : i + window]
        med = np.median(chunk)
        result[i] = np.median(np.abs(chunk - med)) * 1.4826

    return result


def _group_flare_events(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    flux_err: NDArray[np.float64],
    above_threshold: NDArray[np.bool_],
    baseline: NDArray[np.float64],
    sigma_values: NDArray[np.float64],
    min_duration_minutes: float,
    cadence_minutes: float,
) -> list[Flare]:
    """Group consecutive above-threshold points into flare events.

    Args:
        time: Time array
        flux: Flux array
        flux_err: Flux error array
        above_threshold: Boolean mask of points above detection threshold
        baseline: Local baseline flux
        sigma_values: Sigma values at each point
        min_duration_minutes: Minimum flare duration
        cadence_minutes: Data cadence in minutes

    Returns:
        List of Flare objects
    """
    flares: list[Flare] = []

    # Find runs of consecutive detections
    indices = np.where(above_threshold)[0]
    if len(indices) == 0:
        return flares

    # Split into groups (gap > 3 cadences = separate flare)
    groups: list[list[int]] = []
    current_group: list[int] = [indices[0]]

    for i in range(1, len(indices)):
        if indices[i] - indices[i - 1] <= 3:
            current_group.append(indices[i])
        else:
            groups.append(current_group)
            current_group = [indices[i]]
    groups.append(current_group)

    # Convert groups to Flare objects
    for group in groups:
        if len(group) < 2:
            continue

        start_idx = group[0]
        end_idx = group[-1]

        duration_minutes = (time[end_idx] - time[start_idx]) * 24.0 * 60.0

        if duration_minutes < min_duration_minutes:
            continue

        # Find peak within flare
        flare_flux = flux[start_idx : end_idx + 1]
        peak_local_idx = int(np.argmax(flare_flux))
        peak_idx = start_idx + peak_local_idx

        # Compute amplitude (peak - baseline)
        amplitude = float(flux[peak_idx] - baseline[peak_idx])

        # Estimate energy (very rough: proportional to amplitude * duration)
        # Assuming solar luminosity scaling and typical M dwarf values
        # This is a placeholder formula - real energy estimates need stellar params
        energy_estimate = _estimate_flare_energy(
            amplitude, duration_minutes, float(np.mean(flux_err[start_idx : end_idx + 1]))
        )

        flares.append(
            Flare(
                start_time=float(time[start_idx]),
                end_time=float(time[end_idx]),
                peak_time=float(time[peak_idx]),
                amplitude=amplitude,
                duration_minutes=duration_minutes,
                energy_estimate=energy_estimate,
            )
        )

    # Sort by peak time
    flares.sort(key=lambda f: f.peak_time)

    return flares


def _estimate_flare_energy(
    amplitude: float,
    duration_minutes: float,
    flux_err: float,
) -> float:
    """Estimate flare bolometric energy (rough approximation).

    This is a simplified estimate assuming:
    - 10,000 K blackbody flare emission
    - Solar luminosity scaling
    - Triangular flare shape

    Args:
        amplitude: Peak fractional flux increase
        duration_minutes: Total flare duration
        flux_err: Flux uncertainty for signal check

    Returns:
        Estimated energy in ergs (order of magnitude only)
    """
    # Only estimate if amplitude is significant
    if amplitude < 3 * flux_err:
        return 0.0

    # Rough scaling: 1% amplitude for 10 minutes ~ 10^32 erg for M dwarf
    # Scale as amplitude * duration
    duration_hours = duration_minutes / 60.0
    equivalent_duration_seconds = 0.5 * amplitude * duration_hours * 3600.0

    # Assume L_flare ~ 0.01 * L_sun for typical M dwarf flare
    # L_sun = 3.828e33 erg/s
    l_flare = 0.01 * 3.828e33 * amplitude

    energy = l_flare * equivalent_duration_seconds

    return energy


def measure_rotation_period(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    min_period: float = 0.5,
    max_period: float = 30.0,
    n_periods: int = 3000,
) -> tuple[float, float, float]:
    """Measure stellar rotation period with uncertainty estimation.

    Uses Lomb-Scargle periodogram with harmonic detection and uncertainty
    from peak width at half-maximum.

    Args:
        time: Time array, in days (e.g., BTJD)
        flux: Normalized flux array (median ~1.0)
        min_period: Minimum period to search, in days
        max_period: Maximum period to search, in days
        n_periods: Number of period grid points

    Returns:
        Tuple of (period, period_err, snr):
        - period: Best rotation period, in days
        - period_err: Uncertainty on period, in days
        - snr: Detection signal-to-noise ratio
    """
    if len(time) < 100:
        return 1.0, 1.0, 0.0

    # Limit max_period to half the baseline
    baseline = float(time[-1] - time[0])
    max_period = min(max_period, baseline / 2.0)

    if max_period <= min_period:
        return 1.0, 1.0, 0.0

    # Generate period grid (log-spaced for efficiency)
    periods = np.logspace(np.log10(min_period), np.log10(max_period), n_periods, dtype=np.float64)

    # Center flux around zero for Lomb-Scargle
    flux_centered = flux - np.mean(flux)

    # Convert periods to angular frequencies
    angular_frequencies = 2.0 * np.pi / periods

    # Compute Lomb-Scargle periodogram
    power = signal.lombscargle(
        time,
        flux_centered,
        angular_frequencies,
        normalize=True,
    )

    # Find best period
    best_idx = int(np.argmax(power))
    best_period = float(periods[best_idx])
    best_power = float(power[best_idx])

    # Compute SNR using MAD-based noise estimate
    snr = _compute_periodogram_snr(best_power, power)

    # Estimate uncertainty from peak width at half-maximum
    period_err = _estimate_period_uncertainty(periods, power, best_idx)

    # Check for harmonics (period at 2x or 0.5x might be stronger in phase curve)
    # Return the fundamental if harmonics are detected
    if best_period < 2.0:
        # Check if 2x period has similar power (suggests we found first harmonic)
        double_period = best_period * 2.0
        if double_period < max_period:
            double_idx = int(np.argmin(np.abs(periods - double_period)))
            if power[double_idx] > 0.7 * best_power:
                # First harmonic might be the fundamental; keep shorter period
                pass  # Keep best_period as is

    return best_period, period_err, snr


def _compute_periodogram_snr(peak_power: float, power: NDArray[np.float64]) -> float:
    """Compute SNR for periodogram peak.

    Args:
        peak_power: Power at the peak
        power: Full power spectrum

    Returns:
        Signal-to-noise ratio
    """
    median_power = float(np.median(power))
    mad = float(np.median(np.abs(power - median_power)))
    sigma = mad * 1.4826

    if sigma <= 0:
        # Cap at 999.0 to prevent misleading Infinity values for spurious detections
        return 999.0 if peak_power > median_power else 0.0

    # Cap at 999 to prevent misleading extreme values
    return min(999.0, max(0.0, (peak_power - median_power) / sigma))


def _estimate_period_uncertainty(
    periods: NDArray[np.float64],
    power: NDArray[np.float64],
    peak_idx: int,
) -> float:
    """Estimate period uncertainty from peak width at half-maximum.

    Args:
        periods: Period grid
        power: Power spectrum
        peak_idx: Index of the peak

    Returns:
        Period uncertainty in days
    """
    peak_power = power[peak_idx]
    half_max = peak_power / 2.0

    # Search left for half-maximum
    left_idx = peak_idx
    while left_idx > 0 and power[left_idx] > half_max:
        left_idx -= 1

    # Search right for half-maximum
    right_idx = peak_idx
    while right_idx < len(power) - 1 and power[right_idx] > half_max:
        right_idx += 1

    # FWHM in period
    fwhm = abs(periods[right_idx] - periods[left_idx])

    # Uncertainty is approximately FWHM / (2 * sqrt(2 * ln(2)))
    # Simplified to FWHM / 2.355
    uncertainty = fwhm / 2.355

    # Ensure reasonable bounds
    best_period = periods[peak_idx]
    uncertainty = max(uncertainty, 0.001)  # At least 1.4 minutes
    uncertainty = min(uncertainty, best_period * 0.3)  # At most 30% of period

    return float(uncertainty)


def classify_variability(
    periodogram_power: float,
    phase_amplitude: float,
    flare_count: int,
    baseline_days: float,
) -> str:
    """Classify stellar variability type.

    Classification heuristics:
    - spotted_rotator: strong LS peak (SNR > 5), phase amplitude > 1000 ppm
    - pulsator: multiple LS peaks at non-harmonic ratios (not implemented here)
    - eclipsing_binary: V-shaped phase curve, possible secondary (requires phase data)
    - flare_star: high flare rate (>0.5/day)
    - quiet: variability < 500 ppm

    Args:
        periodogram_power: SNR of the strongest periodogram peak
        phase_amplitude: Peak-to-peak phase curve amplitude in fractional flux
        flare_count: Number of detected flares
        baseline_days: Total observation baseline, in days

    Returns:
        Classification string: "spotted_rotator", "pulsator", "eclipsing_binary",
        "flare_star", or "quiet"
    """
    phase_amplitude_ppm = phase_amplitude * 1e6

    # Calculate flare rate
    flare_rate = flare_count / baseline_days if baseline_days > 0 else 0.0

    # Classification logic (in priority order)

    # Flare star: high flare rate
    if flare_rate > 0.5:
        return "flare_star"

    # Quiet star: very low variability
    if phase_amplitude_ppm < 500 and periodogram_power < 3.0:
        return "quiet"

    # Spotted rotator: strong periodic signal
    if periodogram_power > 5.0 and phase_amplitude_ppm > 1000:
        return "spotted_rotator"

    # Default: modest variability but no strong classification
    # Requires both periodogram detection AND meaningful phase amplitude
    if periodogram_power > 3.0 and phase_amplitude_ppm > 500:
        return "spotted_rotator"

    return "quiet"


def compute_activity_index(
    variability_ppm: float,
    rotation_period: float,
    flare_rate: float,
) -> float:
    """Compute photometric activity proxy on 0-1 scale.

    Combines variability amplitude, rotation period (shorter = more active),
    and flare rate into a single activity index.

    Scaling is calibrated such that:
    - Quiet star (Pi Men): ~0.1
    - Moderate activity: ~0.4-0.6
    - Very active star (AU Mic): ~0.8-0.9
    - Extreme activity (Proxima Cen flares): ~0.95

    Args:
        variability_ppm: RMS variability in parts per million
        rotation_period: Rotation period, in days (shorter = more active)
        flare_rate: Flare rate in flares per day

    Returns:
        Activity index between 0.0 and 1.0
    """
    # Component 1: Variability amplitude (log scale, 100-100000 ppm range)
    var_component = np.clip(np.log10(max(variability_ppm, 100)) - 2, 0, 3) / 3.0

    # Component 2: Rotation (shorter period = more active, 1-30 day scale)
    rot_component = 1.0 - np.clip((rotation_period - 1.0) / 29.0, 0, 1)

    # Component 3: Flare rate (log scale, 0.01-10 flares/day)
    flare_component = np.clip(np.log10(max(flare_rate, 0.01)) + 2, 0, 3) / 3.0

    # Weighted combination
    # Variability is most important, then rotation, then flares
    activity_index = 0.5 * var_component + 0.3 * rot_component + 0.2 * flare_component

    # Ensure bounds
    return float(np.clip(activity_index, 0.0, 1.0))


def mask_flares(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    flares: list[Flare],
    buffer_minutes: float = 5.0,
) -> NDArray[np.float64]:
    """Replace flare regions with interpolated baseline.

    Args:
        time: Time array, in days
        flux: Flux array
        flares: List of detected flares
        buffer_minutes: Extra buffer around each flare, in minutes

    Returns:
        Flux array with flares replaced by linear interpolation
    """
    if not flares:
        return flux.copy()

    masked_flux = flux.copy()
    buffer_days = buffer_minutes / (24.0 * 60.0)

    for flare in flares:
        # Expand flare window by buffer
        start = flare.start_time - buffer_days
        end = flare.end_time + buffer_days

        # Find points in flare region
        in_flare = (time >= start) & (time <= end)
        flare_indices = np.where(in_flare)[0]

        if len(flare_indices) == 0:
            continue

        # Get boundary points for interpolation
        first_idx = flare_indices[0]
        last_idx = flare_indices[-1]

        # Get baseline values (before and after flare)
        baseline_before = flux[first_idx - 1] if first_idx > 0 else flux[first_idx]
        baseline_after = flux[last_idx + 1] if last_idx < len(flux) - 1 else flux[last_idx]

        # Linear interpolation
        n_points = len(flare_indices)
        interp_values = np.linspace(baseline_before, baseline_after, n_points)
        masked_flux[flare_indices] = interp_values

    return masked_flux


def compute_phase_amplitude(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    period: float,
    n_bins: int = 20,
) -> float:
    """Compute peak-to-peak phase curve amplitude.

    Args:
        time: Time array, in days
        flux: Flux array
        period: Period for phase folding, in days
        n_bins: Number of phase bins

    Returns:
        Peak-to-peak amplitude (max - min of binned phase curve)
    """
    if len(time) < n_bins:
        return float(np.std(flux))

    # Compute phase
    phase = (time % period) / period

    # Bin by phase
    bin_edges = np.linspace(0, 1, n_bins + 1)
    binned_flux = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (phase >= bin_edges[i]) & (phase < bin_edges[i + 1])
        if np.any(mask):
            binned_flux[i] = np.median(flux[mask])
        else:
            binned_flux[i] = np.nan

    # Remove NaN bins
    valid_bins = ~np.isnan(binned_flux)
    if np.sum(valid_bins) < 3:
        return float(np.std(flux))

    binned_flux = binned_flux[valid_bins]

    return float(np.max(binned_flux) - np.min(binned_flux))


def generate_recommendation(
    variability_class: str,
    variability_ppm: float,
    rotation_period: float,
    flare_rate: float,
    activity_index: float,
    n_expected_transits: int = 10,
    residual_scatter_ppm: float | None = None,
) -> tuple[str, dict[str, Any]]:
    """Generate recommendation for transit detection.

    Args:
        variability_class: Variability classification
        variability_ppm: Variability amplitude in ppm
        rotation_period: Rotation period, in days
        flare_rate: Flare rate in flares per day
        activity_index: Activity index (0-1)
        n_expected_transits: Expected number of transits for depth estimation
        residual_scatter_ppm: Residual scatter after detrending in ppm
            (defaults to variability_ppm / 5 for spotted rotators)

    Returns:
        Tuple of (recommendation_text, suggested_params_dict):
        - recommendation_text: Human-readable recommendation string
        - suggested_params_dict: Machine-actionable parameters for recover_transit
    """
    # Use residual scatter if provided, otherwise estimate it
    if residual_scatter_ppm is None:
        # After detrending, expect ~20% of original variability as residual
        residual_scatter_ppm = variability_ppm / 5.0

    if variability_class == "quiet":
        # Quiet stars: standard tools work, minimal params needed
        min_depth = 100
        params: dict[str, Any] = {
            "min_detectable_depth_ppm": min_depth,
        }
        return (
            "Low stellar activity. Standard transit search tools should work well. "
            f"Minimum detectable depth ~{min_depth} ppm.",
            params,
        )

    if variability_class == "flare_star":
        # Flare stars: need flare masking, depth limited by scatter
        # Use improved formula: 5 * scatter / sqrt(n_transits)
        sqrt_n = np.sqrt(max(n_expected_transits, 1))
        min_depth = int(5.0 * residual_scatter_ppm / sqrt_n)
        params = {
            "rotation_period": round(rotation_period, 2),
            "n_harmonics": 3,
            "min_detectable_depth_ppm": min_depth,
            "use_flare_masking": True,
        }
        return (
            f"High flare activity (rate={flare_rate:.2f}/day). "
            f"Use flare masking before transit search. "
            f"Minimum detectable depth ~{min_depth} ppm due to scatter.",
            params,
        )

    if variability_class == "spotted_rotator":
        # Spotted rotators: need recover_transit with rotation period
        # Use improved formula: 5 * residual_scatter / sqrt(n_transits)
        sqrt_n = np.sqrt(max(n_expected_transits, 1))
        min_depth = int(5.0 * residual_scatter_ppm / sqrt_n)

        # Scale n_harmonics based on activity: higher activity = more harmonics
        n_harmonics = 3 if activity_index < 0.6 else (4 if activity_index < 0.8 else 5)

        params = {
            "rotation_period": round(rotation_period, 2),
            "n_harmonics": n_harmonics,
            "min_detectable_depth_ppm": min_depth,
        }
        return (
            f"High stellar activity. Transit detection requires "
            f"recover_transit with rotation_period={rotation_period:.2f}. "
            f"Minimum detectable depth ~{min_depth} ppm "
            f"(planet > {_depth_to_radius(min_depth / 1e6):.1f} R_Earth).",
            params,
        )

    # Default: moderate activity
    sqrt_n = np.sqrt(max(n_expected_transits, 1))
    min_depth = int(5.0 * residual_scatter_ppm / sqrt_n)
    params = {
        "rotation_period": round(rotation_period, 2),
        "n_harmonics": 3,
        "min_detectable_depth_ppm": min_depth,
    }
    return (
        f"Moderate stellar activity (index={activity_index:.2f}). "
        "Consider using recover_transit for improved sensitivity.",
        params,
    )


def _depth_to_radius(depth: float, r_star: float = 1.0) -> float:
    """Convert transit depth to planet radius.

    Args:
        depth: Fractional transit depth
        r_star: Stellar radius in solar radii (default 1.0)

    Returns:
        Planet radius in Earth radii
    """
    # (R_p/R_star)^2 = depth
    # R_p = R_star * sqrt(depth)
    # Convert to Earth radii (R_sun = 109 R_earth)
    r_sun_to_earth = 109.0
    r_planet = r_star * np.sqrt(depth) * r_sun_to_earth
    return float(r_planet)
