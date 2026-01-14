"""Tests for validation result schema types."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from bittr_tess_vetter.validation.result_schema import (
    CheckResult,
    VettingBundleResult,
    error_result,
    ok_result,
    skipped_result,
)


class TestCheckResult:
    """Tests for CheckResult model."""

    def test_ok_status(self) -> None:
        result = CheckResult(
            id="V01",
            name="Test Check",
            status="ok",
            metrics={"snr": 15.5},
        )
        assert result.status == "ok"
        assert result.id == "V01"

    def test_skipped_status(self) -> None:
        result = CheckResult(
            id="V01",
            name="Test Check",
            status="skipped",
            flags=["SKIPPED:NO_DATA"],
        )
        assert result.status == "skipped"
        assert result.confidence is None

    def test_error_status(self) -> None:
        result = CheckResult(
            id="V01",
            name="Test Check",
            status="error",
            flags=["ERROR:ValueError"],
        )
        assert result.status == "error"

    def test_json_serializable(self) -> None:
        result = CheckResult(
            id="V01",
            name="Test Check",
            status="ok",
            confidence=0.95,
            metrics={"snr": 15.5, "depth_ppm": 1000, "valid": True, "label": "good"},
            flags=["FLAG_A", "FLAG_B"],
            notes=["This is a note"],
            provenance={"version": "0.1.0"},
        )
        # Should not raise
        json_str = result.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["id"] == "V01"
        assert parsed["metrics"]["snr"] == 15.5

    def test_metrics_types(self) -> None:
        """Metrics should accept float, int, str, bool, None."""
        result = CheckResult(
            id="V01",
            name="Test",
            status="ok",
            metrics={
                "float_val": 1.5,
                "int_val": 42,
                "str_val": "test",
                "bool_val": True,
                "none_val": None,
            },
        )
        assert result.metrics["float_val"] == 1.5
        assert result.metrics["int_val"] == 42

    def test_extra_fields_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            CheckResult(
                id="V01",
                name="Test",
                status="ok",
                unknown_field="bad",  # type: ignore[call-arg]
            )


class TestVettingBundleResult:
    """Tests for VettingBundleResult model."""

    def test_empty_bundle(self) -> None:
        bundle = VettingBundleResult()
        assert bundle.results == []
        assert bundle.warnings == []

    def test_bundle_with_results(self) -> None:
        results = [
            CheckResult(id="V01", name="Check 1", status="ok", metrics={"a": 1}),
            CheckResult(id="V02", name="Check 2", status="skipped", flags=["SKIPPED:NO_TPF"]),
        ]
        bundle = VettingBundleResult(
            results=results,
            warnings=["Network disabled"],
            inputs_summary={"has_tpf": False, "network": False},
        )
        assert len(bundle.results) == 2
        assert bundle.results[0].status == "ok"
        assert bundle.results[1].status == "skipped"

    def test_json_serializable(self) -> None:
        bundle = VettingBundleResult(
            results=[CheckResult(id="V01", name="Test", status="ok", metrics={"x": 1})],
            provenance={"pipeline_version": "0.1.0", "duration_ms": 150},
        )
        json_str = bundle.model_dump_json()
        parsed = json.loads(json_str)
        assert len(parsed["results"]) == 1


class TestResultHelpers:
    """Tests for result constructor helpers."""

    def test_ok_result(self) -> None:
        result = ok_result(
            "V01",
            "Odd-Even Depth",
            metrics={"odd_depth": 1000, "even_depth": 1010, "ratio": 0.99},
            confidence=0.95,
            flags=["CONSISTENT"],
        )
        assert result.status == "ok"
        assert result.confidence == 0.95
        assert result.metrics["ratio"] == 0.99
        assert "CONSISTENT" in result.flags

    def test_ok_result_minimal(self) -> None:
        result = ok_result("V01", "Test", metrics={"value": 1})
        assert result.status == "ok"
        assert result.flags == []
        assert result.notes == []

    def test_skipped_result(self) -> None:
        result = skipped_result(
            "V08",
            "Centroid Shift",
            reason_flag="NO_TPF",
            notes=["TPF data not provided"],
        )
        assert result.status == "skipped"
        assert result.confidence is None
        assert result.metrics == {}
        assert "SKIPPED:NO_TPF" in result.flags
        assert "TPF data not provided" in result.notes

    def test_error_result(self) -> None:
        result = error_result(
            "V03",
            "Transit Model Fit",
            error="ValueError",
            notes=["Failed to converge"],
        )
        assert result.status == "error"
        assert "ERROR:ValueError" in result.flags
        assert "Failed to converge" in result.notes

    def test_error_result_with_extra_flags(self) -> None:
        result = error_result(
            "V03",
            "Transit Model Fit",
            error="TimeoutError",
            flags=["BUDGET_EXCEEDED"],
        )
        assert "ERROR:TimeoutError" in result.flags
        assert "BUDGET_EXCEEDED" in result.flags
