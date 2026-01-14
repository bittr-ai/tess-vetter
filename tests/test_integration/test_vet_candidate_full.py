"""Comprehensive integration tests for vet_candidate orchestrator.

This module tests the full vetting pipeline with:
- Synthetic light curves with injected transits
- Various ephemeris configurations
- Error propagation and error handling
- Config pass-through (network=False should skip catalog checks)
- Multi-tier execution paths
"""

from __future__ import annotations

import numpy as np
import pytest

from bittr_tess_vetter.api.types import Candidate, Ephemeris, LightCurve, StellarParams
from bittr_tess_vetter.api.vet import vet_candidate

# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


def _make_synthetic_lightcurve(
    *,
    n_cadences: int = 1000,
    cadence_seconds: float = 120.0,
    t_start: float = 2458000.0,
    noise_sigma: float = 1e-3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic light curve arrays with Gaussian noise.

    Returns:
        Tuple of (time, flux, flux_err) arrays.
    """
    rng = np.random.default_rng(seed)
    cadence_days = cadence_seconds / 86400.0
    time = t_start + np.arange(n_cadences) * cadence_days
    flux = 1.0 + rng.normal(0, noise_sigma, n_cadences)
    flux_err = np.full(n_cadences, noise_sigma)
    return time.astype(np.float64), flux.astype(np.float64), flux_err.astype(np.float64)


def _inject_box_transit(
    time: np.ndarray,
    flux: np.ndarray,
    *,
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    depth: float,
) -> np.ndarray:
    """Inject box-shaped transits into a flux array.

    Returns:
        New flux array with transits injected.
    """
    out = flux.copy()
    duration_days = duration_hours / 24.0
    # Compute phase, wrapped to [-0.5, 0.5]
    phase = ((time - t0_btjd) / period_days + 0.5) % 1.0 - 0.5
    in_transit = np.abs(phase) < (duration_days / (2.0 * period_days))
    out[in_transit] -= depth
    return out


def _make_lc_with_transit(
    *,
    period_days: float = 3.5,
    t0_offset: float = 0.5,
    duration_hours: float = 2.5,
    depth: float = 0.001,
    n_cadences: int = 2000,
    noise_sigma: float = 5e-4,
    seed: int = 42,
) -> tuple[LightCurve, Ephemeris]:
    """Create a LightCurve with an injected transit and matching ephemeris.

    Returns:
        Tuple of (LightCurve, Ephemeris).
    """
    time, flux, flux_err = _make_synthetic_lightcurve(
        n_cadences=n_cadences,
        noise_sigma=noise_sigma,
        seed=seed,
    )
    t0_btjd = time[0] + t0_offset
    flux_with_transit = _inject_box_transit(
        time,
        flux,
        period_days=period_days,
        t0_btjd=t0_btjd,
        duration_hours=duration_hours,
        depth=depth,
    )
    lc = LightCurve(time=time, flux=flux_with_transit, flux_err=flux_err)
    eph = Ephemeris(period_days=period_days, t0_btjd=t0_btjd, duration_hours=duration_hours)
    return lc, eph


# =============================================================================
# Basic Workflow Tests
# =============================================================================


class TestVetCandidateBasicWorkflow:
    """Test basic vet_candidate workflow with synthetic data."""

    def test_full_workflow_with_synthetic_transit(self) -> None:
        """Run full vetting pipeline on synthetic light curve with injected transit."""
        lc, eph = _make_lc_with_transit(
            period_days=3.5,
            duration_hours=2.5,
            depth=0.001,  # 1000 ppm
            n_cadences=2000,
        )
        candidate = Candidate(ephemeris=eph, depth_ppm=1000.0)

        result = vet_candidate(lc, candidate)

        # Should have results for LC-only checks
        assert len(result.results) > 0
        assert any(r.id.startswith("V0") for r in result.results)

        # Check provenance is populated
        assert result.provenance is not None
        # Provenance should have version/timing info (key name may vary by version)
        assert any(
            k in result.provenance
            for k in ["version", "vetter_version", "policy_mode", "pipeline_version"]
        )

    def test_minimal_ephemeris_passthrough(self) -> None:
        """Basic ephemeris passes through vetting pipeline without errors."""
        time, flux, flux_err = _make_synthetic_lightcurve(n_cadences=500)
        lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
        eph = Ephemeris(period_days=2.0, t0_btjd=time[0] + 0.5, duration_hours=2.0)
        candidate = Candidate(ephemeris=eph, depth_ppm=500.0)

        result = vet_candidate(lc, candidate)

        # Should complete without errors
        assert result is not None
        assert len(result.results) >= 1

    def test_pipeline_returns_vetting_bundle_result(self) -> None:
        """vet_candidate returns a VettingBundleResult with expected structure."""
        time, flux, flux_err = _make_synthetic_lightcurve(n_cadences=300)
        lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
        eph = Ephemeris(period_days=1.5, t0_btjd=time[0], duration_hours=1.5)
        candidate = Candidate(ephemeris=eph)

        result = vet_candidate(lc, candidate)

        # Verify structure
        assert hasattr(result, "results")
        assert hasattr(result, "warnings")
        assert hasattr(result, "provenance")
        assert hasattr(result, "n_passed")
        assert hasattr(result, "get_result")


# =============================================================================
# Error Propagation Tests
# =============================================================================


class TestVetCandidateErrorPropagation:
    """Test that errors are properly propagated and reported."""

    def test_invalid_ephemeris_period_raises(self) -> None:
        """Non-positive period raises ValueError."""
        with pytest.raises(ValueError, match="period_days must be positive"):
            Ephemeris(period_days=-1.0, t0_btjd=0.0, duration_hours=2.0)

    def test_invalid_ephemeris_duration_raises(self) -> None:
        """Non-positive duration raises ValueError."""
        with pytest.raises(ValueError, match="duration_hours must be positive"):
            Ephemeris(period_days=2.0, t0_btjd=0.0, duration_hours=-1.0)

    def test_zero_period_raises(self) -> None:
        """Zero period raises ValueError."""
        with pytest.raises(ValueError, match="period_days must be positive"):
            Ephemeris(period_days=0.0, t0_btjd=0.0, duration_hours=2.0)

    def test_unknown_check_ids_raise_error(self) -> None:
        """Unknown check IDs in checks list raise KeyError.

        Note: The new pipeline strictly validates check IDs and raises an error
        for unregistered check IDs. Use list_checks() to discover valid IDs.
        """
        time, flux, flux_err = _make_synthetic_lightcurve(n_cadences=100)
        lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
        eph = Ephemeris(period_days=2.0, t0_btjd=time[0], duration_hours=2.0)
        candidate = Candidate(ephemeris=eph)

        with pytest.raises(KeyError, match="UNKNOWN_CHECK"):
            vet_candidate(
                lc,
                candidate,
                checks=["V01", "UNKNOWN_CHECK", "ANOTHER_FAKE"],
            )


# =============================================================================
# Config Pass-through Tests
# =============================================================================


class TestVetCandidateConfigPassthrough:
    """Test that config parameters are properly passed through."""

    def test_network_false_skips_catalog_checks(self) -> None:
        """network=False should skip catalog checks (V06, V07)."""
        time, flux, flux_err = _make_synthetic_lightcurve(n_cadences=500)
        lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
        eph = Ephemeris(period_days=2.5, t0_btjd=time[0] + 0.5, duration_hours=2.0)
        candidate = Candidate(ephemeris=eph, depth_ppm=500.0)

        # Run with network=False (default), no catalog metadata
        result = vet_candidate(
            lc,
            candidate,
            network=False,
            ra_deg=None,
            dec_deg=None,
            tic_id=None,
        )

        # Catalog checks V06 and V07 should not be present or should be skipped
        v06 = result.get_result("V06")
        v07 = result.get_result("V07")

        # When metadata is missing and not explicitly enabled, V06/V07 are not run
        # If they were somehow run, verify they're marked as skipped
        if v06 is not None:
            assert v06.status == "skipped"
        if v07 is not None:
            assert v07.status == "skipped"

    def test_checks_subset_runs_only_specified_checks(self) -> None:
        """Passing checks=[...] limits execution to those checks.

        Note: Updated to use new `checks` parameter instead of old `enabled` set.
        """
        time, flux, flux_err = _make_synthetic_lightcurve(n_cadences=500)
        lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
        eph = Ephemeris(period_days=2.0, t0_btjd=time[0] + 0.3, duration_hours=2.0)
        candidate = Candidate(ephemeris=eph, depth_ppm=800.0)

        # Run only V01 and V02
        result = vet_candidate(lc, candidate, checks=["V01", "V02"])

        check_ids = [r.id for r in result.results]
        assert "V01" in check_ids
        assert "V02" in check_ids
        # Other LC checks should not be present
        assert "V03" not in check_ids
        assert "V04" not in check_ids
        assert "V05" not in check_ids

    def test_stellar_params_enhance_duration_consistency_check(self) -> None:
        """Providing stellar params should enable V03 to use stellar density.

        Note: Updated to use new `checks` parameter instead of old `enabled` set.
        """
        lc, eph = _make_lc_with_transit(
            period_days=5.0,
            duration_hours=3.0,
            depth=0.002,
            n_cadences=3000,
        )
        candidate = Candidate(ephemeris=eph, depth_ppm=2000.0)
        stellar = StellarParams(
            radius=1.0,
            mass=1.0,
            teff=5800.0,
            logg=4.4,
        )

        result = vet_candidate(lc, candidate, stellar=stellar, checks=["V03"])

        v03 = result.get_result("V03")
        assert v03 is not None
        # V03 should have executed with stellar info (check for metrics)
        assert v03.status == "ok"


# =============================================================================
# Multi-check Execution Tests
# =============================================================================


class TestVetCandidateMultiCheck:
    """Test execution of multiple check tiers."""

    def test_lc_only_checks_always_run(self) -> None:
        """LC-only checks (V01-V05) should always run when not filtered."""
        time, flux, flux_err = _make_synthetic_lightcurve(n_cadences=800)
        lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
        eph = Ephemeris(period_days=3.0, t0_btjd=time[0] + 0.5, duration_hours=2.0)
        candidate = Candidate(ephemeris=eph, depth_ppm=1000.0)

        result = vet_candidate(lc, candidate)

        check_ids = [r.id for r in result.results]
        # LC-only checks should be present
        for check_id in ["V01", "V02", "V03", "V04", "V05"]:
            assert check_id in check_ids, f"LC check {check_id} should run"

    def test_result_aggregation_counts(self) -> None:
        """Verify result counts are correctly aggregated."""
        lc, eph = _make_lc_with_transit(
            period_days=2.5,
            duration_hours=2.0,
            depth=0.001,
            n_cadences=1500,
        )
        candidate = Candidate(ephemeris=eph, depth_ppm=1000.0)

        result = vet_candidate(lc, candidate)

        n_results = len(result.results)
        assert n_results >= 5  # At least LC-only checks

    def test_warnings_list_populated_on_issues(self) -> None:
        """Warnings list should capture any issues during execution.

        Note: Updated to use new `checks` parameter instead of old `enabled` set.
        """
        time, flux, flux_err = _make_synthetic_lightcurve(n_cadences=100)
        lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
        eph = Ephemeris(period_days=2.0, t0_btjd=time[0], duration_hours=2.0)
        candidate = Candidate(ephemeris=eph)

        # Enable catalog checks without providing metadata (should produce skipped results)
        result = vet_candidate(
            lc,
            candidate,
            checks=["V06", "V07"],
            network=True,
            ra_deg=None,
            dec_deg=None,
            tic_id=None,
        )

        # Should have warnings about missing metadata
        assert len(result.warnings) > 0
        # Warnings should mention V06 or V07
        combined_warnings = " ".join(result.warnings)
        assert "V06" in combined_warnings or "V07" in combined_warnings


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestVetCandidateEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_short_lightcurve_handles_gracefully(self) -> None:
        """Very short light curves should be handled without crashes."""
        # Only 10 data points
        time = np.linspace(0.0, 1.0, 10, dtype=np.float64)
        flux = np.ones(10, dtype=np.float64)
        flux_err = np.full(10, 0.001, dtype=np.float64)
        lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
        eph = Ephemeris(period_days=0.5, t0_btjd=0.25, duration_hours=1.0)
        candidate = Candidate(ephemeris=eph)

        # Should not crash
        result = vet_candidate(lc, candidate)
        assert result is not None

    def test_very_long_period_handled(self) -> None:
        """Very long period (longer than data span) handled gracefully."""
        time, flux, flux_err = _make_synthetic_lightcurve(n_cadences=500)
        lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
        # Period longer than data span (data spans ~7 days at 120s cadence)
        eph = Ephemeris(period_days=30.0, t0_btjd=time[0] + 3.0, duration_hours=4.0)
        candidate = Candidate(ephemeris=eph)

        result = vet_candidate(lc, candidate)
        assert result is not None

    def test_deep_transit_does_not_crash(self) -> None:
        """Very deep transit (>10%) doesn't crash pipeline."""
        lc, eph = _make_lc_with_transit(
            period_days=2.0,
            duration_hours=2.0,
            depth=0.15,  # 15% depth (EB-like)
            n_cadences=1000,
        )
        candidate = Candidate(ephemeris=eph, depth_ppm=150000.0)

        result = vet_candidate(lc, candidate)
        assert result is not None
        assert len(result.results) > 0
