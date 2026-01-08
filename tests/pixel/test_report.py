"""Tests for bittr_tess_vetter.pixel.report module.

Comprehensive tests for the pixel vetting report generator including:
- PixelVetReport model validation
- generate_pixel_vet_report function
- Threshold version handling
- Individual test evaluation
- Pass/fail determination logic
"""

from __future__ import annotations

import numpy as np
import pytest

from bittr_tess_vetter.pixel.aperture import ApertureDependenceResult
from bittr_tess_vetter.pixel.centroid import CentroidResult
from bittr_tess_vetter.pixel.difference import DifferenceImageResult
from bittr_tess_vetter.pixel.report import (
    THRESHOLD_VERSIONS,
    PixelVetReport,
    generate_pixel_vet_report,
)

# =============================================================================
# Test Fixtures - Helper functions for building test data
# =============================================================================


def make_passing_centroid_result() -> CentroidResult:
    """Create a CentroidResult that should pass all tests.

    Low centroid shift and low significance.
    """
    return CentroidResult(
        centroid_shift_pixels=0.3,  # Below 1.0 threshold
        significance_sigma=1.5,  # Below 3.0 threshold
        in_transit_centroid=(5.0, 5.0),
        out_of_transit_centroid=(5.2, 5.1),
        n_in_transit_cadences=20,
        n_out_transit_cadences=80,
    )


def make_failing_centroid_shift() -> CentroidResult:
    """Create a CentroidResult that fails the shift test."""
    return CentroidResult(
        centroid_shift_pixels=1.5,  # Above 1.0 threshold
        significance_sigma=1.5,  # Below 3.0 threshold (passes)
        in_transit_centroid=(5.0, 5.0),
        out_of_transit_centroid=(6.2, 5.5),
        n_in_transit_cadences=20,
        n_out_transit_cadences=80,
    )


def make_failing_centroid_significance() -> CentroidResult:
    """Create a CentroidResult that fails the significance test."""
    return CentroidResult(
        centroid_shift_pixels=0.5,  # Below 1.0 threshold (passes)
        significance_sigma=4.5,  # Above 3.0 threshold
        in_transit_centroid=(5.0, 5.0),
        out_of_transit_centroid=(5.3, 5.2),
        n_in_transit_cadences=20,
        n_out_transit_cadences=80,
    )


def make_passing_diff_image_result() -> DifferenceImageResult:
    """Create a DifferenceImageResult that should pass all tests."""
    return DifferenceImageResult(
        difference_image=np.zeros((11, 11)),
        localization_score=0.9,  # Above 0.7 threshold
        brightest_pixel_coords=(5, 5),
        target_coords=(5, 5),
        distance_to_target=0.0,
    )


def make_failing_diff_image_result() -> DifferenceImageResult:
    """Create a DifferenceImageResult that fails the localization test."""
    return DifferenceImageResult(
        difference_image=np.zeros((11, 11)),
        localization_score=0.5,  # Below 0.7 threshold
        brightest_pixel_coords=(0, 0),
        target_coords=(5, 5),
        distance_to_target=7.07,
    )


def make_passing_aperture_result() -> ApertureDependenceResult:
    """Create an ApertureDependenceResult that should pass all tests."""
    return ApertureDependenceResult(
        depths_by_aperture={1.0: 1000.0, 2.0: 1010.0, 3.0: 995.0},
        stability_metric=0.95,  # Above 0.8 threshold
        recommended_aperture=2.0,
        depth_variance=56.25,
    )


def make_failing_aperture_result() -> ApertureDependenceResult:
    """Create an ApertureDependenceResult that fails the stability test."""
    return ApertureDependenceResult(
        depths_by_aperture={1.0: 500.0, 2.0: 1500.0, 3.0: 800.0},
        stability_metric=0.5,  # Below 0.8 threshold
        recommended_aperture=2.0,
        depth_variance=170555.56,
    )


# =============================================================================
# THRESHOLD_VERSIONS Tests
# =============================================================================


class TestThresholdVersions:
    """Tests for THRESHOLD_VERSIONS configuration."""

    def test_v1_exists(self) -> None:
        """v1 threshold version exists."""
        assert "v1" in THRESHOLD_VERSIONS

    def test_v1_has_required_keys(self) -> None:
        """v1 has all required threshold keys."""
        v1 = THRESHOLD_VERSIONS["v1"]
        required_keys = {
            "max_centroid_shift_pixels",
            "min_centroid_significance_sigma",
            "min_localization_score",
            "min_aperture_stability",
        }
        assert set(v1.keys()) == required_keys

    def test_v1_values_are_reasonable(self) -> None:
        """v1 threshold values are reasonable."""
        v1 = THRESHOLD_VERSIONS["v1"]

        # Centroid shift should be positive and not too large
        assert 0 < v1["max_centroid_shift_pixels"] <= 5.0

        # Significance threshold should be positive
        assert v1["min_centroid_significance_sigma"] > 0

        # Localization score should be between 0 and 1
        assert 0 < v1["min_localization_score"] < 1

        # Aperture stability should be between 0 and 1
        assert 0 < v1["min_aperture_stability"] < 1


# =============================================================================
# generate_pixel_vet_report Basic Tests
# =============================================================================


class TestGeneratePixelVetReportBasic:
    """Basic tests for generate_pixel_vet_report function."""

    def test_generates_valid_report(self) -> None:
        """generate_pixel_vet_report returns a valid PixelVetReport."""
        centroid = make_passing_centroid_result()
        diff_image = make_passing_diff_image_result()
        aperture = make_passing_aperture_result()

        report = generate_pixel_vet_report(
            centroid_result=centroid,
            diff_image_result=diff_image,
            aperture_result=aperture,
        )

        assert isinstance(report, PixelVetReport)

    def test_stores_input_results(self) -> None:
        """Report stores the input result objects."""
        centroid = make_passing_centroid_result()
        diff_image = make_passing_diff_image_result()
        aperture = make_passing_aperture_result()

        report = generate_pixel_vet_report(
            centroid_result=centroid,
            diff_image_result=diff_image,
            aperture_result=aperture,
        )

        assert report.centroid_result == centroid
        assert report.diff_image_result == diff_image
        assert report.aperture_result == aperture

    def test_accepts_plots(self) -> None:
        """Report stores plots metadata when provided."""
        centroid = make_passing_centroid_result()
        diff_image = make_passing_diff_image_result()
        aperture = make_passing_aperture_result()
        plots = [{"kind": "centroid"}, {"kind": "difference_image"}]

        report = generate_pixel_vet_report(
            centroid_result=centroid,
            diff_image_result=diff_image,
            aperture_result=aperture,
            plots=plots,
        )

        assert report.plots == plots

    def test_defaults_to_empty_plots(self) -> None:
        """Report defaults to empty plots list."""
        centroid = make_passing_centroid_result()
        diff_image = make_passing_diff_image_result()
        aperture = make_passing_aperture_result()

        report = generate_pixel_vet_report(
            centroid_result=centroid,
            diff_image_result=diff_image,
            aperture_result=aperture,
        )

        assert report.plots == []

    def test_defaults_to_empty_plots_repeated(self) -> None:
        """Report defaults to empty plots list."""
        centroid = make_passing_centroid_result()
        diff_image = make_passing_diff_image_result()
        aperture = make_passing_aperture_result()

        report = generate_pixel_vet_report(
            centroid_result=centroid,
            diff_image_result=diff_image,
            aperture_result=aperture,
        )

        assert report.plots == []

    def test_defaults_to_empty_plots_again(self) -> None:
        """Report defaults to empty plots list."""
        centroid = make_passing_centroid_result()
        diff_image = make_passing_diff_image_result()
        aperture = make_passing_aperture_result()

        report = generate_pixel_vet_report(
            centroid_result=centroid,
            diff_image_result=diff_image,
            aperture_result=aperture,
        )

        assert report.plots == []


# =============================================================================
# generate_pixel_vet_report Pass/Fail Tests
# =============================================================================


class TestGeneratePixelVetReportPassFail:
    """Tests for pass/fail determination in generate_pixel_vet_report."""

    def test_all_passing_yields_pixel_pass_true(self) -> None:
        """All tests passing yields pixel_pass=True."""
        centroid = make_passing_centroid_result()
        diff_image = make_passing_diff_image_result()
        aperture = make_passing_aperture_result()

        report = generate_pixel_vet_report(
            centroid_result=centroid,
            diff_image_result=diff_image,
            aperture_result=aperture,
        )

        assert report.pixel_pass is True
        assert len(report.failure_reasons) == 0

    def test_centroid_shift_fail_yields_pixel_pass_false(self) -> None:
        """Failing centroid shift test yields pixel_pass=False."""
        centroid = make_failing_centroid_shift()
        diff_image = make_passing_diff_image_result()
        aperture = make_passing_aperture_result()

        report = generate_pixel_vet_report(
            centroid_result=centroid,
            diff_image_result=diff_image,
            aperture_result=aperture,
        )

        assert report.pixel_pass is False
        assert report.individual_tests["centroid_shift"] is False
        assert len(report.failure_reasons) >= 1

    def test_centroid_significance_fail_yields_pixel_pass_false(self) -> None:
        """Failing centroid significance test yields pixel_pass=False."""
        centroid = make_failing_centroid_significance()
        diff_image = make_passing_diff_image_result()
        aperture = make_passing_aperture_result()

        report = generate_pixel_vet_report(
            centroid_result=centroid,
            diff_image_result=diff_image,
            aperture_result=aperture,
        )

        assert report.pixel_pass is False
        assert report.individual_tests["centroid_significance"] is False
        assert len(report.failure_reasons) >= 1

    def test_localization_fail_yields_pixel_pass_false(self) -> None:
        """Failing localization test yields pixel_pass=False."""
        centroid = make_passing_centroid_result()
        diff_image = make_failing_diff_image_result()
        aperture = make_passing_aperture_result()

        report = generate_pixel_vet_report(
            centroid_result=centroid,
            diff_image_result=diff_image,
            aperture_result=aperture,
        )

        assert report.pixel_pass is False
        assert report.individual_tests["localization"] is False
        assert len(report.failure_reasons) >= 1

    def test_aperture_stability_fail_yields_pixel_pass_false(self) -> None:
        """Failing aperture stability test yields pixel_pass=False."""
        centroid = make_passing_centroid_result()
        diff_image = make_passing_diff_image_result()
        aperture = make_failing_aperture_result()

        report = generate_pixel_vet_report(
            centroid_result=centroid,
            diff_image_result=diff_image,
            aperture_result=aperture,
        )

        assert report.pixel_pass is False
        assert report.individual_tests["aperture_stability"] is False
        assert len(report.failure_reasons) >= 1

    def test_multiple_failures_accumulated(self) -> None:
        """Multiple test failures are all recorded."""
        centroid = make_failing_centroid_shift()
        diff_image = make_failing_diff_image_result()
        aperture = make_failing_aperture_result()

        report = generate_pixel_vet_report(
            centroid_result=centroid,
            diff_image_result=diff_image,
            aperture_result=aperture,
        )

        assert report.pixel_pass is False
        # At least 3 failures (shift, localization, aperture_stability)
        assert len(report.failure_reasons) >= 3


# =============================================================================
# generate_pixel_vet_report Individual Tests
# =============================================================================


class TestGeneratePixelVetReportIndividualTests:
    """Tests for individual test recording in generate_pixel_vet_report."""

    def test_individual_tests_has_all_keys(self) -> None:
        """individual_tests dict has all expected keys."""
        centroid = make_passing_centroid_result()
        diff_image = make_passing_diff_image_result()
        aperture = make_passing_aperture_result()

        report = generate_pixel_vet_report(
            centroid_result=centroid,
            diff_image_result=diff_image,
            aperture_result=aperture,
        )

        expected_keys = {
            "centroid_shift",
            "centroid_significance",
            "localization",
            "aperture_stability",
        }
        assert set(report.individual_tests.keys()) == expected_keys

    def test_all_individual_tests_pass(self) -> None:
        """All individual tests pass with good inputs."""
        centroid = make_passing_centroid_result()
        diff_image = make_passing_diff_image_result()
        aperture = make_passing_aperture_result()

        report = generate_pixel_vet_report(
            centroid_result=centroid,
            diff_image_result=diff_image,
            aperture_result=aperture,
        )

        assert all(report.individual_tests.values())


# =============================================================================
# generate_pixel_vet_report Threshold Version Tests
# =============================================================================


class TestGeneratePixelVetReportThresholdVersion:
    """Tests for threshold version handling in generate_pixel_vet_report."""

    def test_uses_v1_by_default(self) -> None:
        """Default threshold version is v1."""
        centroid = make_passing_centroid_result()
        diff_image = make_passing_diff_image_result()
        aperture = make_passing_aperture_result()

        # This should not raise
        report = generate_pixel_vet_report(
            centroid_result=centroid,
            diff_image_result=diff_image,
            aperture_result=aperture,
        )

        assert isinstance(report, PixelVetReport)

    def test_accepts_explicit_v1(self) -> None:
        """Explicit v1 threshold version is accepted."""
        centroid = make_passing_centroid_result()
        diff_image = make_passing_diff_image_result()
        aperture = make_passing_aperture_result()

        report = generate_pixel_vet_report(
            centroid_result=centroid,
            diff_image_result=diff_image,
            aperture_result=aperture,
            threshold_version="v1",
        )

        assert isinstance(report, PixelVetReport)

    def test_rejects_unknown_version(self) -> None:
        """Unknown threshold version raises ValueError."""
        centroid = make_passing_centroid_result()
        diff_image = make_passing_diff_image_result()
        aperture = make_passing_aperture_result()

        with pytest.raises(ValueError) as exc_info:
            generate_pixel_vet_report(
                centroid_result=centroid,
                diff_image_result=diff_image,
                aperture_result=aperture,
                threshold_version="v99",
            )

        assert "v99" in str(exc_info.value)
        assert "Valid versions" in str(exc_info.value)


# =============================================================================
# generate_pixel_vet_report Failure Reason Tests
# =============================================================================


class TestGeneratePixelVetReportFailureReasons:
    """Tests for failure reason messages in generate_pixel_vet_report."""

    def test_centroid_shift_failure_message(self) -> None:
        """Centroid shift failure includes informative message."""
        centroid = make_failing_centroid_shift()
        diff_image = make_passing_diff_image_result()
        aperture = make_passing_aperture_result()

        report = generate_pixel_vet_report(
            centroid_result=centroid,
            diff_image_result=diff_image,
            aperture_result=aperture,
        )

        # Find the centroid shift failure reason
        shift_reasons = [r for r in report.failure_reasons if "Centroid shift" in r]
        assert len(shift_reasons) >= 1
        assert "1.5" in shift_reasons[0] or "pixels" in shift_reasons[0]

    def test_localization_failure_message(self) -> None:
        """Localization failure includes informative message."""
        centroid = make_passing_centroid_result()
        diff_image = make_failing_diff_image_result()
        aperture = make_passing_aperture_result()

        report = generate_pixel_vet_report(
            centroid_result=centroid,
            diff_image_result=diff_image,
            aperture_result=aperture,
        )

        # Find the localization failure reason
        loc_reasons = [r for r in report.failure_reasons if "Localization" in r]
        assert len(loc_reasons) >= 1
        assert "0.5" in loc_reasons[0] or "target" in loc_reasons[0]

    def test_aperture_stability_failure_message(self) -> None:
        """Aperture stability failure includes informative message."""
        centroid = make_passing_centroid_result()
        diff_image = make_passing_diff_image_result()
        aperture = make_failing_aperture_result()

        report = generate_pixel_vet_report(
            centroid_result=centroid,
            diff_image_result=diff_image,
            aperture_result=aperture,
        )

        # Find the aperture failure reason
        ap_reasons = [r for r in report.failure_reasons if "Aperture" in r or "stability" in r]
        assert len(ap_reasons) >= 1
        assert "aperture" in ap_reasons[0].lower()


# =============================================================================
# PixelVetReport Model Tests
# =============================================================================


class TestPixelVetReportModel:
    """Tests for PixelVetReport Pydantic model."""

    def test_is_frozen(self) -> None:
        """PixelVetReport is immutable."""
        centroid = make_passing_centroid_result()
        diff_image = make_passing_diff_image_result()
        aperture = make_passing_aperture_result()

        report = generate_pixel_vet_report(
            centroid_result=centroid,
            diff_image_result=diff_image,
            aperture_result=aperture,
        )

        with pytest.raises(Exception):  # ValidationError or AttributeError
            report.pixel_pass = False  # type: ignore

    def test_can_serialize_to_dict(self) -> None:
        """PixelVetReport can be serialized to dict."""
        centroid = make_passing_centroid_result()
        diff_image = make_passing_diff_image_result()
        aperture = make_passing_aperture_result()

        report = generate_pixel_vet_report(
            centroid_result=centroid,
            diff_image_result=diff_image,
            aperture_result=aperture,
        )

        # model_dump should work
        data = report.model_dump()
        assert isinstance(data, dict)
        assert "pixel_pass" in data
        assert "individual_tests" in data


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestGeneratePixelVetReportEdgeCases:
    """Edge case tests for generate_pixel_vet_report."""

    def test_boundary_centroid_shift_at_threshold(self) -> None:
        """Centroid shift exactly at threshold fails."""
        centroid = CentroidResult(
            centroid_shift_pixels=1.0,  # Exactly at 1.0 threshold
            significance_sigma=1.5,
            in_transit_centroid=(5.0, 5.0),
            out_of_transit_centroid=(5.7, 5.7),
            n_in_transit_cadences=20,
            n_out_transit_cadences=80,
        )
        diff_image = make_passing_diff_image_result()
        aperture = make_passing_aperture_result()

        report = generate_pixel_vet_report(
            centroid_result=centroid,
            diff_image_result=diff_image,
            aperture_result=aperture,
        )

        # At threshold should fail (>= comparison)
        assert report.individual_tests["centroid_shift"] is False

    def test_boundary_localization_at_threshold(self) -> None:
        """Localization score exactly at threshold passes."""
        centroid = make_passing_centroid_result()
        diff_image = DifferenceImageResult(
            difference_image=np.zeros((11, 11)),
            localization_score=0.7,  # Exactly at 0.7 threshold
            brightest_pixel_coords=(5, 5),
            target_coords=(5, 5),
            distance_to_target=0.0,
        )
        aperture = make_passing_aperture_result()

        report = generate_pixel_vet_report(
            centroid_result=centroid,
            diff_image_result=diff_image,
            aperture_result=aperture,
        )

        # At threshold should pass (>= comparison)
        assert report.individual_tests["localization"] is True

    def test_boundary_aperture_stability_at_threshold(self) -> None:
        """Aperture stability exactly at threshold passes."""
        centroid = make_passing_centroid_result()
        diff_image = make_passing_diff_image_result()
        aperture = ApertureDependenceResult(
            depths_by_aperture={1.0: 1000.0, 2.0: 1010.0, 3.0: 995.0},
            stability_metric=0.8,  # Exactly at 0.8 threshold
            recommended_aperture=2.0,
            depth_variance=56.25,
        )

        report = generate_pixel_vet_report(
            centroid_result=centroid,
            diff_image_result=diff_image,
            aperture_result=aperture,
        )

        # At threshold should pass (>= comparison)
        assert report.individual_tests["aperture_stability"] is True

    def test_just_below_centroid_shift_threshold_passes(self) -> None:
        """Centroid shift just below threshold passes."""
        centroid = CentroidResult(
            centroid_shift_pixels=0.999,  # Just below 1.0 threshold
            significance_sigma=1.5,
            in_transit_centroid=(5.0, 5.0),
            out_of_transit_centroid=(5.7, 5.7),
            n_in_transit_cadences=20,
            n_out_transit_cadences=80,
        )
        diff_image = make_passing_diff_image_result()
        aperture = make_passing_aperture_result()

        report = generate_pixel_vet_report(
            centroid_result=centroid,
            diff_image_result=diff_image,
            aperture_result=aperture,
        )

        assert report.individual_tests["centroid_shift"] is True


# =============================================================================
# Integration Tests
# =============================================================================


class TestGeneratePixelVetReportIntegration:
    """Integration tests for complete pixel vetting workflow."""

    def test_full_passing_workflow(self) -> None:
        """Complete workflow with all passing tests."""
        centroid = make_passing_centroid_result()
        diff_image = make_passing_diff_image_result()
        aperture = make_passing_aperture_result()

        report = generate_pixel_vet_report(
            centroid_result=centroid,
            diff_image_result=diff_image,
            aperture_result=aperture,
            threshold_version="v1",
            plots=[{"kind": "centroid"}, {"kind": "diff"}, {"kind": "aperture"}],
        )

        # All assertions for a complete passing report
        assert report.pixel_pass is True
        assert all(report.individual_tests.values())
        assert len(report.failure_reasons) == 0
        assert len(report.plots) == 3
        assert report.centroid_result == centroid
        assert report.diff_image_result == diff_image
        assert report.aperture_result == aperture

    def test_full_failing_workflow(self) -> None:
        """Complete workflow with all failing tests."""
        centroid = CentroidResult(
            centroid_shift_pixels=2.0,  # Fails shift
            significance_sigma=5.0,  # Fails significance
            in_transit_centroid=(5.0, 5.0),
            out_of_transit_centroid=(7.0, 5.0),
            n_in_transit_cadences=20,
            n_out_transit_cadences=80,
        )
        diff_image = make_failing_diff_image_result()
        aperture = make_failing_aperture_result()

        report = generate_pixel_vet_report(
            centroid_result=centroid,
            diff_image_result=diff_image,
            aperture_result=aperture,
        )

        # All assertions for a complete failing report
        assert report.pixel_pass is False
        assert not any(
            [
                report.individual_tests["centroid_shift"],
                report.individual_tests["centroid_significance"],
                report.individual_tests["localization"],
                report.individual_tests["aperture_stability"],
            ]
        )
        assert len(report.failure_reasons) == 4
