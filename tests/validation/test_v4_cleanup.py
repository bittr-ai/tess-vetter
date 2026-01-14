"""Tests for v4 cleanup: no stubs exported, no default warnings.

This module verifies:
1. Stub symbols (check_nearby_eb_search, check_known_fp_match, etc.) are NOT exported
2. vet_candidate() does not emit warnings by default
3. Deprecated params still emit warnings (backward compatibility)

Novelty: standard (testing cleanup completeness)
"""

from __future__ import annotations

import warnings as python_warnings

import numpy as np
import pytest

from bittr_tess_vetter.api import (
    Candidate,
    Ephemeris,
    LightCurve,
    vet_candidate,
)

# =============================================================================
# Test 1: No stub symbols exported from validation module
# =============================================================================


def test_stub_symbols_not_exported_from_validation() -> None:
    """Stub functions (V06-V10) should NOT be exported from validation module."""
    from bittr_tess_vetter import validation

    # These stub symbols should NOT be in __all__
    stub_symbols = [
        "check_nearby_eb_search",
        "check_known_fp_match",
        "check_centroid_shift",
        "check_pixel_level_lc",
        "check_aperture_dependence",
        "run_all_checks",
    ]

    for symbol in stub_symbols:
        assert symbol not in validation.__all__, (
            f"Stub symbol '{symbol}' should not be in validation.__all__"
        )


def test_stub_symbols_not_importable_from_validation() -> None:
    """Stub functions should not be importable from validation module."""
    stub_symbols = [
        "check_nearby_eb_search",
        "check_known_fp_match",
        "check_centroid_shift",
        "check_pixel_level_lc",
        "check_aperture_dependence",
        "run_all_checks",
    ]

    from bittr_tess_vetter import validation

    for symbol in stub_symbols:
        assert not hasattr(validation, symbol), (
            f"Stub symbol '{symbol}' should not be accessible from validation module"
        )


def test_lc_checks_only_contains_v01_v05() -> None:
    """lc_checks module should only contain V01-V05 implementations."""
    from bittr_tess_vetter.validation import lc_checks

    # V01-V05 should be present
    expected_functions = [
        "check_odd_even_depth",
        "check_secondary_eclipse",
        "check_duration_consistency",
        "check_depth_stability",
        "check_v_shape",
    ]

    for func_name in expected_functions:
        assert hasattr(lc_checks, func_name), (
            f"LC check function '{func_name}' should be present in lc_checks"
        )

    # Stub functions should NOT be present
    stub_functions = [
        "check_nearby_eb_search",
        "check_known_fp_match",
        "check_centroid_shift",
        "check_pixel_level_lc",
        "check_aperture_dependence",
        "run_all_checks",
    ]

    for func_name in stub_functions:
        assert not hasattr(lc_checks, func_name), (
            f"Stub function '{func_name}' should not be present in lc_checks"
        )


# =============================================================================
# Test 2: No default warnings from vet_candidate
# =============================================================================


@pytest.fixture
def synthetic_lc() -> LightCurve:
    """Create a synthetic light curve for testing."""
    np.random.seed(42)
    time = np.linspace(0, 27, 2000)
    flux = np.ones_like(time) + np.random.normal(0, 0.001, len(time))
    flux_err = np.ones_like(flux) * 0.001
    return LightCurve(time=time, flux=flux, flux_err=flux_err)


@pytest.fixture
def synthetic_candidate() -> Candidate:
    """Create a synthetic candidate for testing."""
    return Candidate(
        ephemeris=Ephemeris(
            period_days=3.5,
            t0_btjd=1.0,
            duration_hours=2.4,
        ),
    )


def test_vet_candidate_no_warnings_by_default(
    synthetic_lc: LightCurve, synthetic_candidate: Candidate
) -> None:
    """vet_candidate should not emit warnings by default."""
    with python_warnings.catch_warnings(record=True) as w:
        python_warnings.simplefilter("always")

        # Run vet_candidate with typical skip scenario
        result = vet_candidate(
            synthetic_lc,
            synthetic_candidate,
            network=False,  # Should skip catalog checks
            tpf=None,  # Should skip pixel checks
        )

        # Filter out unrelated warnings (from numpy, etc.)
        bittr_warnings = [warning for warning in w if "bittr_tess_vetter" in str(warning.filename)]

        # Should not emit any bittr-tess-vetter warnings
        assert len(bittr_warnings) == 0, (
            f"vet_candidate emitted unexpected warnings: {[str(w.message) for w in bittr_warnings]}"
        )

    # Verify structural skip info in results
    assert result is not None
    assert len(result.results) > 0


def test_skipped_checks_have_structured_info(
    synthetic_lc: LightCurve, synthetic_candidate: Candidate
) -> None:
    """Skipped checks should have structured info, not warnings."""
    result = vet_candidate(
        synthetic_lc,
        synthetic_candidate,
        network=False,
        tpf=None,
    )

    # Find skipped checks
    skipped_checks = [r for r in result.results if r.status == "skipped"]

    # Each skipped check should have a reason flag
    for check in skipped_checks:
        assert len(check.flags) > 0, (
            f"Skipped check {check.id} should have at least one flag explaining why"
        )


# =============================================================================
# Test 3: Deprecated params still emit warnings (backward compat)
# =============================================================================


def test_deprecated_policy_mode_emits_warning() -> None:
    """Using deprecated policy_mode should emit FutureWarning."""
    from bittr_tess_vetter.api.lc_only import odd_even_depth

    np.random.seed(42)
    time = np.linspace(0, 27, 2000)
    flux = np.ones_like(time) + np.random.normal(0, 0.001, len(time))
    lc = LightCurve(time=time, flux=flux, flux_err=np.ones_like(flux) * 0.001)
    eph = Ephemeris(period_days=3.5, t0_btjd=1.0, duration_hours=2.4)

    with python_warnings.catch_warnings(record=True) as w:
        python_warnings.simplefilter("always")

        # Using non-default policy_mode should warn
        odd_even_depth(lc, eph, policy_mode="strict")  # type: ignore[arg-type]

        # Should have emitted a FutureWarning
        future_warnings = [warning for warning in w if issubclass(warning.category, FutureWarning)]
        assert len(future_warnings) >= 1, "Using deprecated policy_mode should emit FutureWarning"


# =============================================================================
# Test 4: No "deferred" or "stub" status in check results
# =============================================================================


def test_no_deferred_status_in_results(
    synthetic_lc: LightCurve, synthetic_candidate: Candidate
) -> None:
    """Check results should never have 'deferred' status - use 'skipped' with reason."""
    result = vet_candidate(
        synthetic_lc,
        synthetic_candidate,
        network=False,
        tpf=None,
    )

    for check_result in result.results:
        # Status should never be "deferred"
        assert check_result.status != "deferred", (
            f"Check {check_result.id} returned deprecated 'deferred' status"
        )

        # Metrics should not contain "deferred" or "stub" keys
        metrics = check_result.metrics or {}
        assert "deferred" not in metrics, f"Check {check_result.id} has 'deferred' in metrics"
        assert "stub" not in metrics, f"Check {check_result.id} has 'stub' in metrics"


def test_no_deferred_flags_in_results(
    synthetic_lc: LightCurve, synthetic_candidate: Candidate
) -> None:
    """Flags should not contain 'deferred' - use descriptive skip reasons."""
    result = vet_candidate(
        synthetic_lc,
        synthetic_candidate,
        network=False,
        tpf=None,
    )

    for check_result in result.results:
        for flag in check_result.flags:
            assert "deferred" not in flag.lower(), (
                f"Check {check_result.id} has 'deferred' in flags: {flag}"
            )
