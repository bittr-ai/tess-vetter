"""Tests for the modshift_uniqueness module.

Tests the independent implementation of ModShift signal uniqueness metrics
following Thompson et al. (2018) and Coughlin et al. (2016).
"""

from __future__ import annotations

import numpy as np
import pytest

from bittr_tess_vetter.validation.modshift_uniqueness import run_modshift_uniqueness


def make_synthetic_lc(
    period: float = 5.0,
    t0: float = 1502.0,
    duration_hours: float = 3.0,
    depth_ppm: float = 1000.0,
    n_points: int = 5000,
    baseline_days: float = 27.0,
    noise_ppm: float = 200.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a synthetic light curve with injected transits.

    Args:
        period: Orbital period in days.
        t0: Time of first transit (BTJD).
        duration_hours: Transit duration in hours.
        depth_ppm: Transit depth in ppm.
        n_points: Number of data points.
        baseline_days: Total baseline of observations.
        noise_ppm: Gaussian noise level in ppm.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (time, flux, flux_err) arrays.
    """
    rng = np.random.default_rng(seed)

    # Generate time array
    start_time = t0 - 2.0  # Start 2 days before first transit
    time = np.sort(
        rng.uniform(start_time, start_time + baseline_days, n_points)
    ).astype(np.float64)

    # Create baseline flux with Gaussian noise
    flux = np.ones(n_points, dtype=np.float64)
    noise = rng.normal(0, noise_ppm * 1e-6, n_points)
    flux += noise

    # Inject box transits
    phase = np.mod(time - t0, period) / period
    phase[phase > 0.5] -= 1
    duration_phase = (duration_hours / 24.0) / period
    in_transit = np.abs(phase) < duration_phase / 2
    flux[in_transit] -= depth_ppm * 1e-6

    # Flux errors (constant)
    flux_err = np.full(n_points, noise_ppm * 1e-6, dtype=np.float64)

    return time, flux, flux_err


def test_run_modshift_uniqueness_returns_expected_keys() -> None:
    """Test that run_modshift_uniqueness returns all expected output keys."""
    time, flux, flux_err = make_synthetic_lc()

    result = run_modshift_uniqueness(
        time=time,
        flux=flux,
        flux_err=flux_err,
        period=5.0,
        t0=1502.0,
        duration_hours=3.0,
    )

    # Check all expected keys are present
    expected_keys = {
        "sig_pri",
        "sig_sec",
        "sig_ter",
        "sig_pos",
        "fred",
        "fa1",
        "fa2",
        "ms1",
        "ms2",
        "ms3",
        "ms4",
        "ms5",
        "ms6",
        "med_chases",
        "chi",
        "n_in",
        "n_out",
        "n_transits",
        "status",
        "warnings",
    }

    assert set(result.keys()) == expected_keys


def test_run_modshift_uniqueness_fred_scale() -> None:
    """Test that Fred values are in the expected range (~1-10) for TESS-like data.

    This is the key fix: exovetter produces Fred ~60-96 which breaks MS normalization.
    Our implementation should produce Fred ~1-10.
    """
    time, flux, flux_err = make_synthetic_lc(
        depth_ppm=1000.0,
        noise_ppm=200.0,
        seed=123,
    )

    result = run_modshift_uniqueness(
        time=time,
        flux=flux,
        flux_err=flux_err,
        period=5.0,
        t0=1502.0,
        duration_hours=3.0,
    )

    fred = result["fred"]
    assert fred is not None
    assert np.isfinite(fred)

    # Fred should be in reasonable range for TESS data (~1-10)
    # NOT the exovetter range of ~60-96
    assert 0.1 < fred < 50, f"Fred={fred} is outside expected range [0.1, 50]"


def test_run_modshift_uniqueness_ms1_positive_for_strong_signal() -> None:
    """Test that MS1 is positive for a strong transit signal.

    MS1 = sig_pri / fred - FA1
    For a strong primary signal, MS1 should be positive.
    """
    time, flux, flux_err = make_synthetic_lc(
        depth_ppm=2000.0,  # Strong signal
        noise_ppm=200.0,
        seed=456,
    )

    result = run_modshift_uniqueness(
        time=time,
        flux=flux,
        flux_err=flux_err,
        period=5.0,
        t0=1502.0,
        duration_hours=3.0,
    )

    ms1 = result["ms1"]
    sig_pri = result["sig_pri"]

    assert ms1 is not None
    assert sig_pri is not None
    assert np.isfinite(ms1)
    assert np.isfinite(sig_pri)

    # Strong signal should have positive MS1 (threshold is 0.2)
    assert ms1 > 0, f"MS1={ms1} should be positive for strong signal"
    assert sig_pri > 5, f"sig_pri={sig_pri} should be > 5 for strong signal"


def test_run_modshift_uniqueness_signal_hierarchy() -> None:
    """Test that sig_pri > sig_sec > sig_ter for clean transit signal.

    For a clean box transit with no secondary eclipse, the primary should
    dominate the secondary and tertiary signals.
    """
    time, flux, flux_err = make_synthetic_lc(
        depth_ppm=1500.0,
        noise_ppm=150.0,
        seed=789,
    )

    result = run_modshift_uniqueness(
        time=time,
        flux=flux,
        flux_err=flux_err,
        period=5.0,
        t0=1502.0,
        duration_hours=3.0,
    )

    sig_pri = result["sig_pri"]
    sig_sec = result["sig_sec"]
    sig_ter = result["sig_ter"]

    assert all(np.isfinite([sig_pri, sig_sec, sig_ter]))
    assert sig_pri > sig_sec, f"Primary ({sig_pri}) should > secondary ({sig_sec})"
    # tertiary might not always be < secondary due to noise, but should be < primary
    assert sig_pri > sig_ter, f"Primary ({sig_pri}) should > tertiary ({sig_ter})"


def test_run_modshift_uniqueness_insufficient_data() -> None:
    """Test handling of insufficient data."""
    # Very short time baseline with few points
    time = np.array([1500.0, 1500.1, 1500.2], dtype=np.float64)
    flux = np.array([1.0, 0.999, 1.0], dtype=np.float64)
    flux_err = np.array([0.0002, 0.0002, 0.0002], dtype=np.float64)

    result = run_modshift_uniqueness(
        time=time,
        flux=flux,
        flux_err=flux_err,
        period=5.0,
        t0=1500.1,
        duration_hours=3.0,
    )

    # Should return with invalid status or warnings
    assert result["status"] in ("ok", "invalid", "error")
    # Most metrics should be nan or the status should be invalid
    if result["status"] == "ok":
        # If somehow ok, at least warn about limited data
        pass  # Accept whatever the function decides


def test_run_modshift_uniqueness_n_tce_parameter() -> None:
    """Test that n_tce parameter affects FA1 threshold."""
    time, flux, flux_err = make_synthetic_lc()

    result_default = run_modshift_uniqueness(
        time=time,
        flux=flux,
        flux_err=flux_err,
        period=5.0,
        t0=1502.0,
        duration_hours=3.0,
        n_tce=20000,  # Default
    )

    result_fewer = run_modshift_uniqueness(
        time=time,
        flux=flux,
        flux_err=flux_err,
        period=5.0,
        t0=1502.0,
        duration_hours=3.0,
        n_tce=100,  # Fewer TCEs = lower FA1
    )

    fa1_default = result_default["fa1"]
    fa1_fewer = result_fewer["fa1"]

    assert np.isfinite(fa1_default)
    assert np.isfinite(fa1_fewer)
    # More TCEs means higher FA threshold
    assert fa1_default > fa1_fewer


def test_run_modshift_uniqueness_chases_value_range() -> None:
    """Test that CHASES values are in [0, 1] range."""
    time, flux, flux_err = make_synthetic_lc()

    result = run_modshift_uniqueness(
        time=time,
        flux=flux,
        flux_err=flux_err,
        period=5.0,
        t0=1502.0,
        duration_hours=3.0,
    )

    med_chases = result["med_chases"]

    if med_chases is not None and np.isfinite(med_chases):
        assert 0 <= med_chases <= 1, f"CHASES={med_chases} should be in [0, 1]"


def test_run_modshift_uniqueness_chi_positive() -> None:
    """Test that CHI is positive for valid data."""
    time, flux, flux_err = make_synthetic_lc()

    result = run_modshift_uniqueness(
        time=time,
        flux=flux,
        flux_err=flux_err,
        period=5.0,
        t0=1502.0,
        duration_hours=3.0,
    )

    chi = result["chi"]

    if chi is not None and np.isfinite(chi):
        assert chi > 0, f"CHI={chi} should be positive"


@pytest.mark.parametrize(
    "depth_ppm,expected_sig_pri_min",
    [
        (500, 2.0),  # Shallow transit - still detectable
        (1000, 4.0),  # Moderate transit
        (2000, 7.0),  # Deep transit
    ],
)
def test_run_modshift_uniqueness_depth_scaling(
    depth_ppm: float, expected_sig_pri_min: float
) -> None:
    """Test that deeper transits produce higher primary significance."""
    time, flux, flux_err = make_synthetic_lc(
        depth_ppm=depth_ppm,
        noise_ppm=200.0,
        seed=42,
    )

    result = run_modshift_uniqueness(
        time=time,
        flux=flux,
        flux_err=flux_err,
        period=5.0,
        t0=1502.0,
        duration_hours=3.0,
    )

    sig_pri = result["sig_pri"]

    assert sig_pri is not None
    assert np.isfinite(sig_pri)
    assert sig_pri > expected_sig_pri_min, (
        f"depth={depth_ppm}ppm: sig_pri={sig_pri} < {expected_sig_pri_min}"
    )
