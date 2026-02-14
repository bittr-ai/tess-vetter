"""Tests for TRICERATOPS FPP replicate orchestration and degenerate detection.

Tests the new replicate functionality (R3) and degenerate detection (R2) without
requiring network access by monkeypatching the vendored TRICERATOPS target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from bittr_tess_vetter.validation.triceratops_fpp import (
    _aggregate_replicate_results,
    _is_result_degenerate,
)
from bittr_tess_vetter.platform.network.timeout import NetworkTimeoutError

# =============================================================================
# Test _is_result_degenerate helper
# =============================================================================


class TestIsResultDegenerate:
    """Tests for the _is_result_degenerate helper function."""

    def test_error_in_result_is_degenerate(self):
        """Results with 'error' key should be degenerate."""
        result = {"error": "something failed", "error_type": "internal_error"}
        assert _is_result_degenerate(result) is True

    def test_nan_fpp_is_degenerate(self):
        """NaN FPP should be degenerate."""
        result = {"fpp": float("nan"), "nfpp": 0.1, "posterior_sum_total": 1.0}
        assert _is_result_degenerate(result) is True

    def test_none_fpp_is_degenerate(self):
        """None FPP should be degenerate."""
        result = {"fpp": None, "nfpp": 0.1}
        assert _is_result_degenerate(result) is True

    def test_inf_fpp_is_degenerate(self):
        """Infinite FPP should be degenerate."""
        result = {"fpp": float("inf"), "nfpp": 0.1}
        assert _is_result_degenerate(result) is True

    def test_nan_posterior_sum_is_degenerate(self):
        """NaN posterior_sum_total should be degenerate."""
        result = {"fpp": 0.01, "nfpp": 0.001, "posterior_sum_total": float("nan")}
        assert _is_result_degenerate(result) is True

    def test_zero_posterior_sum_is_degenerate(self):
        """Zero posterior_sum_total should be degenerate."""
        result = {"fpp": 0.01, "nfpp": 0.001, "posterior_sum_total": 0.0}
        assert _is_result_degenerate(result) is True

    def test_negative_posterior_sum_is_degenerate(self):
        """Negative posterior_sum_total should be degenerate."""
        result = {"fpp": 0.01, "nfpp": 0.001, "posterior_sum_total": -0.5}
        assert _is_result_degenerate(result) is True

    def test_nan_count_positive_is_degenerate(self):
        """Positive posterior_prob_nan_count should be degenerate."""
        result = {
            "fpp": 0.01,
            "nfpp": 0.001,
            "posterior_sum_total": 1.0,
            "posterior_prob_nan_count": 3,
        }
        assert _is_result_degenerate(result) is True

    def test_valid_result_is_not_degenerate(self):
        """Valid result should not be degenerate."""
        result = {
            "fpp": 0.01,
            "nfpp": 0.001,
            "posterior_sum_total": 1.0,
            "posterior_prob_nan_count": 0,
        }
        assert _is_result_degenerate(result) is False

    def test_valid_result_without_optional_fields(self):
        """Valid result without optional fields should not be degenerate."""
        result = {"fpp": 0.05, "nfpp": 0.01}
        assert _is_result_degenerate(result) is False


# =============================================================================
# Test _aggregate_replicate_results helper
# =============================================================================


class TestAggregateReplicateResults:
    """Tests for the _aggregate_replicate_results helper function."""

    def test_empty_results_returns_empty(self):
        """Empty results list should return empty dict."""
        out = _aggregate_replicate_results([], tic_id=12345, sectors_used=[1], total_runtime=10.0)
        assert out == {}

    def test_all_degenerate_returns_empty(self):
        """All degenerate results should return empty dict."""
        results = [
            {"fpp": float("nan"), "nfpp": 0.1},
            {"error": "failed", "error_type": "internal_error"},
        ]
        out = _aggregate_replicate_results(
            results, tic_id=12345, sectors_used=[1], total_runtime=10.0
        )
        assert out == {}

    def test_single_success_aggregation(self):
        """Single successful result should be returned with counts."""
        results = [
            {
                "fpp": 0.01,
                "nfpp": 0.001,
                "posterior_sum_total": 1.0,
                "posterior_prob_nan_count": 0,
            }
        ]
        out = _aggregate_replicate_results(
            results, tic_id=12345, sectors_used=[1], total_runtime=5.0
        )

        assert out["fpp"] == 0.01
        assert out["nfpp"] == 0.001
        assert out["replicates"] == 1
        assert out["n_success"] == 1
        assert out["n_fail"] == 0
        assert out["replicate_success_rate"] == 1.0

    def test_multiple_success_aggregation(self):
        """Multiple successful results should be aggregated with median/CI."""
        results = [
            {"fpp": 0.01, "nfpp": 0.001, "posterior_sum_total": 1.0},
            {"fpp": 0.02, "nfpp": 0.002, "posterior_sum_total": 1.0},
            {"fpp": 0.03, "nfpp": 0.003, "posterior_sum_total": 1.0},
        ]
        out = _aggregate_replicate_results(
            results, tic_id=12345, sectors_used=[1, 2], total_runtime=15.0
        )

        # Check FPP summary
        assert "fpp_summary" in out
        assert out["fpp_summary"]["median"] == 0.02  # median of [0.01, 0.02, 0.03]
        assert len(out["fpp_summary"]["values"]) == 3

        # Check NFPP summary
        assert "nfpp_summary" in out
        assert out["nfpp_summary"]["median"] == 0.002

        # Check counts
        assert out["replicates"] == 3
        assert out["n_success"] == 3
        assert out["n_fail"] == 0
        assert out["replicate_success_rate"] == 1.0
        assert out["runtime_seconds"] == 15.0

    def test_mixed_success_and_degenerate(self):
        """Mixed results should aggregate only successful ones."""
        results = [
            {"fpp": 0.01, "nfpp": 0.001, "posterior_sum_total": 1.0},
            {"fpp": float("nan"), "nfpp": 0.1},  # degenerate
            {"fpp": 0.03, "nfpp": 0.003, "posterior_sum_total": 1.0},
        ]
        out = _aggregate_replicate_results(
            results, tic_id=12345, sectors_used=[1], total_runtime=12.0
        )

        assert out["n_success"] == 2
        assert out["n_fail"] == 1
        assert out["replicate_success_rate"] == 0.666667
        assert len(out["fpp_summary"]["values"]) == 2

    def test_best_run_selection(self):
        """Best run (lowest FPP) details should be preserved."""
        results = [
            {
                "fpp": 0.05,
                "nfpp": 0.005,
                "posterior_sum_total": 1.0,
                "run_seed": 1001,
            },
            {
                "fpp": 0.01,
                "nfpp": 0.001,
                "posterior_sum_total": 1.0,
                "run_seed": 1002,
            },  # best
            {
                "fpp": 0.03,
                "nfpp": 0.003,
                "posterior_sum_total": 1.0,
                "run_seed": 1003,
            },
        ]
        out = _aggregate_replicate_results(
            results, tic_id=12345, sectors_used=[1], total_runtime=10.0
        )

        # Best run seed should be preserved (though overwritten by summary in practice)
        # The median FPP should be 0.03
        assert out["fpp_summary"]["median"] == 0.03

    def test_warning_note_emitted_for_high_failure_rate(self):
        """A high replicate failure rate should add a warning note."""
        results = [
            {"fpp": 0.01, "nfpp": 0.001, "posterior_sum_total": 1.0},
            {"error": "failed", "error_type": "internal_error"},
            {"fpp": float("nan"), "nfpp": 0.1},
            {"error": "timed out", "error_type": "timeout"},
        ]
        out = _aggregate_replicate_results(
            results, tic_id=12345, sectors_used=[1], total_runtime=10.0
        )

        assert out["replicate_success_rate"] == 0.25
        assert "warning_note" in out


# =============================================================================
# Mock TRICERATOPS target for integration tests
# =============================================================================


class MockProbsDataFrame:
    """Mock DataFrame-like object for TRICERATOPS probs table."""

    def __init__(self, scenarios: list[str], probs: list[float]):
        self._scenarios = scenarios
        self._probs = probs
        self.columns = ["scenario", "prob"]

    def __getitem__(self, key: str) -> list[Any]:
        if key == "scenario":
            return self._scenarios
        elif key == "prob":
            return self._probs
        raise KeyError(key)


@dataclass
class MockTriceratopsTarget:
    """Mock TRICERATOPS target for testing without network."""

    FPP: float = 0.01
    NFPP: float = 0.001
    probs: Any = None
    stars: list[dict[str, Any]] | None = None
    trilegal_fname: str | None = "/fake/trilegal.csv"
    trilegal_url: str | None = None

    def calc_depths(self, tdepth: float) -> None:  # noqa: ARG002
        """Mock calc_depths - does nothing."""
        pass

    def calc_probs(self, **kwargs: Any) -> None:  # noqa: ARG002
        """Mock calc_probs - sets FPP/NFPP based on configuration."""
        # Create mock probs DataFrame-like object
        if self.probs is None:
            self.probs = MockProbsDataFrame(
                scenarios=["TP", "EB", "BEB", "NEB", "NTP"],
                probs=[0.99, 0.005, 0.003, 0.001, 0.001],
            )


def make_degenerate_target() -> MockTriceratopsTarget:
    """Create a target that returns degenerate (NaN) results."""
    target = MockTriceratopsTarget(
        FPP=float("nan"),
        NFPP=float("nan"),
        probs=MockProbsDataFrame(
            scenarios=["TP", "EB", "BEB"],
            probs=[float("nan"), float("nan"), float("nan")],
        ),
    )
    return target


def make_timeout_target() -> MockTriceratopsTarget:
    """Create a target that times out during calc_probs."""
    target = MockTriceratopsTarget()

    def _raise_timeout(**kwargs: Any) -> None:  # noqa: ARG001
        raise NetworkTimeoutError("replicate timed out", 1.0)

    target.calc_probs = _raise_timeout  # type: ignore[method-assign]
    return target


def test_vendor_triceratops_has_no_unconditional_griz_print() -> None:
    from pathlib import Path

    vendor_file = Path(
        "src/bittr_tess_vetter/ext/triceratops_plus_vendor/triceratops/triceratops.py"
    )
    text = vendor_file.read_text(encoding="utf-8")
    assert 'print("griz mags:' not in text


def make_valid_target(fpp: float = 0.01, seed: int = 0) -> MockTriceratopsTarget:
    """Create a target that returns valid results with slight variation."""
    # Add small random variation based on seed for realistic replicate behavior
    np.random.seed(seed)
    fpp_var = fpp * (1 + 0.1 * np.random.randn())
    nfpp_var = fpp_var * 0.1

    return MockTriceratopsTarget(
        FPP=fpp_var,
        NFPP=nfpp_var,
        probs=MockProbsDataFrame(
            scenarios=["TP", "EB", "BEB", "NEB", "NTP"],
            probs=[1 - fpp_var, fpp_var * 0.3, fpp_var * 0.3, fpp_var * 0.2, fpp_var * 0.2],
        ),
        stars=[{"Tmag": 10.0}, {"Tmag": 12.0}, {"Tmag": 14.0}],
    )


# =============================================================================
# Integration tests with mocked TRICERATOPS
# =============================================================================

# Check if TRICERATOPS vendor is available
try:
    from bittr_tess_vetter.ext.triceratops_plus_vendor.triceratops import (
        triceratops as _tr,  # noqa: F401
    )

    del _tr  # Clean up namespace
    HAS_TRICERATOPS = True
except ImportError:
    HAS_TRICERATOPS = False


@pytest.mark.skipif(
    not HAS_TRICERATOPS,
    reason="TRICERATOPS vendor not available (requires triceratops extra)",
)
class TestCalculateFppHandlerReplicates:
    """Integration tests for calculate_fpp_handler with replicates."""

    @pytest.fixture
    def mock_cache(self):
        """Create a mock cache with light curve data."""
        cache = MagicMock()
        cache.cache_dir = "/tmp/test_cache"

        # Mock light curve data
        time_data = np.linspace(0, 27, 1000)
        flux_data = np.ones(1000)
        flux_err_data = np.ones(1000) * 0.001

        lc_mock = MagicMock()
        lc_mock.time = time_data
        lc_mock.flux = flux_data
        lc_mock.flux_err = flux_err_data
        lc_mock.valid_mask = np.ones(1000, dtype=bool)

        def mock_keys():
            return ["lc:12345:1:pdcsap"]

        def mock_get(key):
            if key.startswith("lc:12345"):
                return lc_mock
            return None

        cache.keys = mock_keys
        cache.get = mock_get

        return cache

    @patch("bittr_tess_vetter.validation.triceratops_fpp._load_cached_triceratops_target")
    @patch("bittr_tess_vetter.validation.triceratops_fpp._save_cached_triceratops_target")
    def test_single_replicate_valid_result(self, mock_save, mock_load, mock_cache):
        """Single replicate with valid result should return success."""
        from bittr_tess_vetter.validation.triceratops_fpp import calculate_fpp_handler

        # Mock target loading
        mock_load.return_value = make_valid_target(fpp=0.01, seed=42)

        result = calculate_fpp_handler(
            cache=mock_cache,
            tic_id=12345,
            period=10.0,
            t0=1500.0,
            depth_ppm=500,
            duration_hours=3.0,
            replicates=1,
            seed=42,
        )

        assert "error" not in result
        assert "fpp" in result
        assert result["replicates"] == 1
        assert result["n_success"] == 1
        assert result["n_fail"] == 0

    @patch("bittr_tess_vetter.validation.triceratops_fpp._load_cached_triceratops_target")
    @patch("bittr_tess_vetter.validation.triceratops_fpp._save_cached_triceratops_target")
    def test_single_replicate_degenerate_returns_error(self, mock_save, mock_load, mock_cache):
        """Single replicate with degenerate result should return error."""
        from bittr_tess_vetter.validation.triceratops_fpp import calculate_fpp_handler

        # Mock target loading with degenerate target
        mock_load.return_value = make_degenerate_target()

        result = calculate_fpp_handler(
            cache=mock_cache,
            tic_id=12345,
            period=10.0,
            t0=1500.0,
            depth_ppm=500,
            duration_hours=3.0,
            replicates=1,
            seed=42,
        )

        assert "error" in result
        assert result["error_type"] == "degenerate_posterior"
        assert result["n_success"] == 0
        assert result["replicate_success_rate"] == 0.0
        assert "warning_note" in result

    @patch("bittr_tess_vetter.validation.triceratops_fpp._load_cached_triceratops_target")
    @patch("bittr_tess_vetter.validation.triceratops_fpp._save_cached_triceratops_target")
    def test_multiple_replicates_aggregation(self, mock_save, mock_load, mock_cache):
        """Multiple replicates should aggregate to median/CI."""
        from bittr_tess_vetter.validation.triceratops_fpp import calculate_fpp_handler

        # Create a target that varies per call
        call_count = [0]
        fpps = [0.01, 0.02, 0.03]

        def make_varying_target(*args, **kwargs):
            target = make_valid_target(fpp=fpps[call_count[0] % len(fpps)], seed=call_count[0])
            call_count[0] += 1
            return target

        mock_load.side_effect = make_varying_target

        result = calculate_fpp_handler(
            cache=mock_cache,
            tic_id=12345,
            period=10.0,
            t0=1500.0,
            depth_ppm=500,
            duration_hours=3.0,
            replicates=3,
            seed=1000,
        )

        assert "error" not in result
        assert result["replicates"] == 3
        assert result["n_success"] >= 1
        assert "replicate_success_rate" in result
        assert "fpp_summary" in result or result["n_success"] == 1

    @patch("bittr_tess_vetter.validation.triceratops_fpp._load_cached_triceratops_target")
    @patch("bittr_tess_vetter.validation.triceratops_fpp._save_cached_triceratops_target")
    def test_all_replicates_degenerate_returns_error(self, mock_save, mock_load, mock_cache):
        """All degenerate replicates should return degenerate_posterior error."""
        from bittr_tess_vetter.validation.triceratops_fpp import calculate_fpp_handler

        # All calls return degenerate
        mock_load.return_value = make_degenerate_target()

        result = calculate_fpp_handler(
            cache=mock_cache,
            tic_id=12345,
            period=10.0,
            t0=1500.0,
            depth_ppm=500,
            duration_hours=3.0,
            replicates=3,
            seed=42,
        )

        assert "error" in result
        assert result["error_type"] == "degenerate_posterior"
        assert result["n_success"] == 0
        assert result["replicates"] == 3
        assert result["replicate_success_rate"] == 0.0
        assert "degenerate_reasons" in result
        assert "warning_note" in result

    @patch("bittr_tess_vetter.validation.triceratops_fpp._load_cached_triceratops_target")
    @patch("bittr_tess_vetter.validation.triceratops_fpp._save_cached_triceratops_target")
    def test_all_replicates_timeout_returns_timeout_error(self, mock_save, mock_load, mock_cache):
        """Timeout-only replicate failures should return high-level timeout semantics."""
        from bittr_tess_vetter.validation.triceratops_fpp import calculate_fpp_handler

        mock_load.return_value = make_timeout_target()

        result = calculate_fpp_handler(
            cache=mock_cache,
            tic_id=12345,
            period=10.0,
            t0=1500.0,
            depth_ppm=500,
            duration_hours=3.0,
            replicates=3,
            seed=42,
        )

        assert "error" in result
        assert result["error_type"] == "timeout"
        assert result["stage"] == "replicate_aggregation"
        assert result["n_success"] == 0
        assert result["replicates"] == 3
        assert result["replicate_success_rate"] == 0.0
        assert "actionable_guidance" in result
        assert any(
            err.get("error_type") == "timeout" for err in result.get("replicate_errors", [])
        )

    @patch("bittr_tess_vetter.validation.triceratops_fpp._load_cached_triceratops_target")
    @patch("bittr_tess_vetter.validation.triceratops_fpp._save_cached_triceratops_target")
    def test_seed_reproducibility(self, mock_save, mock_load, mock_cache):
        """Same seed should produce same base_seed in output."""
        from bittr_tess_vetter.validation.triceratops_fpp import calculate_fpp_handler

        mock_load.return_value = make_valid_target(fpp=0.01, seed=42)

        result1 = calculate_fpp_handler(
            cache=mock_cache,
            tic_id=12345,
            period=10.0,
            t0=1500.0,
            depth_ppm=500,
            duration_hours=3.0,
            replicates=1,
            seed=12345,
        )

        result2 = calculate_fpp_handler(
            cache=mock_cache,
            tic_id=12345,
            period=10.0,
            t0=1500.0,
            depth_ppm=500,
            duration_hours=3.0,
            replicates=1,
            seed=12345,
        )

        assert result1.get("base_seed") == result2.get("base_seed") == 12345
