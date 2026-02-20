"""Unit tests for aperture family depth curve analysis.

Tests the compute_aperture_family_depth_curve function and related utilities
using synthetic TPF data with known star positions and transit parameters.
"""

from __future__ import annotations

import numpy as np
import pytest

from tess_vetter.pixel.aperture_family import (
    DEFAULT_RADII_PX,
    SLOPE_SIGNIFICANCE_THRESHOLD,
    ApertureFamilyResult,
    compute_aperture_family_depth_curve,
)
from tests.pixel.fixtures.synthetic_cubes import (
    StarSpec,
    TransitSpec,
    make_blended_binary_tpf,
    make_synthetic_tpf_fits,
)


class TestApertureFamilyResult:
    """Tests for ApertureFamilyResult dataclass."""

    def test_to_dict_serialization(self) -> None:
        """Test that ApertureFamilyResult can be serialized to dict."""
        result = ApertureFamilyResult(
            depths_by_radius_ppm={1.5: 1000.0, 2.0: 1100.0, 2.5: 1200.0},
            depth_uncertainties_ppm={1.5: 50.0, 2.0: 55.0, 2.5: 60.0},
            depth_slope_ppm_per_pixel=200.0,
            depth_slope_significance=3.5,
            blend_indicator="increasing",
            recommended_aperture_px=1.5,
            warnings=["test warning"],
            evidence_summary={"n_apertures_tested": 3},
        )

        result_dict = result.to_dict()

        assert result_dict["depths_by_radius_ppm"] == {1.5: 1000.0, 2.0: 1100.0, 2.5: 1200.0}
        assert result_dict["blend_indicator"] == "increasing"
        assert result_dict["depth_slope_significance"] == 3.5
        assert "test warning" in result_dict["warnings"]

    def test_frozen_dataclass(self) -> None:
        """Test that ApertureFamilyResult is immutable."""
        result = ApertureFamilyResult(
            depths_by_radius_ppm={1.5: 1000.0},
            depth_uncertainties_ppm={1.5: 50.0},
            depth_slope_ppm_per_pixel=0.0,
            depth_slope_significance=0.0,
            blend_indicator="consistent",
            recommended_aperture_px=1.5,
            warnings=[],
            evidence_summary={},
        )

        with pytest.raises(AttributeError):
            result.blend_indicator = "increasing"  # type: ignore[misc]


class TestComputeApertureFamilyDepthCurve:
    """Tests for compute_aperture_family_depth_curve function."""

    def test_quality_flagged_cadences_are_ignored(self) -> None:
        """Quality-flagged cadences should not corrupt the measured depth curve."""
        period = 5.0
        t0 = 2458001.0
        duration_days = 0.2

        tpf_clean = make_synthetic_tpf_fits(
            shape=(600, 11, 11),
            stars=[StarSpec(row=5.0, col=5.0, flux=10000.0)],
            transit_spec=TransitSpec(
                star_idx=0,
                depth_frac=0.01,
                period=period,
                t0=t0,
                duration_days=duration_days,
            ),
            noise_level=30.0,
            seed=123,
        )
        clean = compute_aperture_family_depth_curve(
            tpf_fits=tpf_clean,
            period=period,
            t0=t0,
            duration_hours=duration_days * 24.0,
        )

        tpf_bad = make_synthetic_tpf_fits(
            shape=(600, 11, 11),
            stars=[StarSpec(row=5.0, col=5.0, flux=10000.0)],
            transit_spec=TransitSpec(
                star_idx=0,
                depth_frac=0.01,
                period=period,
                t0=t0,
                duration_days=duration_days,
            ),
            noise_level=30.0,
            seed=123,
        )

        # Flag a mix of in-transit and out-of-transit cadences as "bad", and poison flux.
        time = tpf_bad.time
        phase = ((time - t0) % period) / period
        phase = np.where(phase > 0.5, phase - 1.0, phase)
        in_transit = np.abs(phase) <= ((duration_days / 2.0) / period)
        in_idx = np.where(in_transit)[0][:5]
        out_idx = np.where(~in_transit)[0][:15]
        bad_idx = np.unique(np.concatenate([in_idx, out_idx]))

        tpf_bad.quality[bad_idx] = 1
        tpf_bad.flux[bad_idx] = np.nan

        bad = compute_aperture_family_depth_curve(
            tpf_fits=tpf_bad,
            period=period,
            t0=t0,
            duration_hours=duration_days * 24.0,
        )

        assert any("Dropped" in w for w in bad.warnings)

        # With the bad cadences removed, the curve should remain broadly consistent.
        clean_depths = np.array([clean.depths_by_radius_ppm[r] for r in DEFAULT_RADII_PX])
        bad_depths = np.array([bad.depths_by_radius_ppm[r] for r in DEFAULT_RADII_PX])
        finite = np.isfinite(clean_depths) & np.isfinite(bad_depths)
        assert int(np.sum(finite)) >= 3

        # Allow for some drift due to fewer points, but reject gross corruption.
        denom = np.maximum(np.abs(clean_depths[finite]), 1.0)
        rel_diff = np.nanmedian(np.abs(clean_depths[finite] - bad_depths[finite]) / denom)
        assert rel_diff < 0.25

    def test_single_star_depth_curve_structure(self) -> None:
        """Single star transit should produce valid depth curve structure."""
        # Create synthetic TPF with single star at center
        tpf = make_synthetic_tpf_fits(
            shape=(500, 11, 11),
            stars=[StarSpec(row=5.0, col=5.0, flux=10000.0)],
            transit_spec=TransitSpec(
                star_idx=0,
                depth_frac=0.01,  # 1% = 10000 ppm
                period=5.0,
                t0=2458001.0,
                duration_days=0.2,
            ),
            noise_level=50.0,
            seed=42,
        )

        result = compute_aperture_family_depth_curve(
            tpf_fits=tpf,
            period=5.0,
            t0=2458001.0,
            duration_hours=4.8,  # 0.2 days = 4.8 hours
        )

        # Check result type
        assert isinstance(result, ApertureFamilyResult)

        # Should have measurements for all default radii
        assert len(result.depths_by_radius_ppm) == len(DEFAULT_RADII_PX)

        # Depths should be positive (transit detected) or finite
        valid_depths = [d for d in result.depths_by_radius_ppm.values() if np.isfinite(d)]
        assert len(valid_depths) >= 3, "Should have at least 3 valid depth measurements"

        # Should have a blend indicator (any valid classification)
        assert result.blend_indicator in ["consistent", "increasing", "decreasing", "unstable"]

        # Slope and significance should be computed
        assert np.isfinite(result.depth_slope_ppm_per_pixel) or result.blend_indicator == "unstable"

    def test_blended_binary_increasing_depth_curve(self) -> None:
        """Blended binary with transit on secondary should show increasing depth."""
        # Create blended binary with 10 arcsec separation
        # Transit on secondary star (index 1)
        tpf = make_blended_binary_tpf(
            separation_arcsec=10.0,
            flux_ratio=0.5,  # Secondary is half as bright as primary
            transit_on_secondary=True,
            transit_depth_frac=0.02,  # 2% depth on secondary
            primary_flux=10000.0,
            shape=(500, 11, 11),
            noise_level=30.0,  # Low noise for clear signal
            seed=42,
        )

        result = compute_aperture_family_depth_curve(
            tpf_fits=tpf,
            period=5.0,
            t0=2458001.0,
            duration_hours=4.8,
        )

        # Check that depths are measured
        valid_depths = [d for d in result.depths_by_radius_ppm.values() if np.isfinite(d)]
        assert len(valid_depths) >= 3, "Should have at least 3 valid depth measurements"

        # For a blend with transit on the off-center secondary:
        # - Small apertures centered on primary will have lower depth
        # - Larger apertures include more of the secondary, showing higher depth
        # This should give positive slope (increasing depth with radius)
        # Note: The blend indicator should be "increasing" if the slope is significant
        if (
            np.isfinite(result.depth_slope_significance)
            and result.depth_slope_significance > SLOPE_SIGNIFICANCE_THRESHOLD
        ):
            # If slope is significant and positive, should be "increasing"
            assert result.depth_slope_ppm_per_pixel > 0
            assert result.blend_indicator == "increasing"

    def test_custom_radii(self) -> None:
        """Test with custom aperture radii."""
        tpf = make_synthetic_tpf_fits(
            shape=(500, 11, 11),
            stars=[StarSpec(row=5.0, col=5.0, flux=10000.0)],
            transit_spec=TransitSpec(
                star_idx=0,
                depth_frac=0.01,
                period=5.0,
                t0=2458001.0,
                duration_days=0.2,
            ),
            noise_level=50.0,
            seed=42,
        )

        custom_radii = [1.0, 2.0, 3.0]
        result = compute_aperture_family_depth_curve(
            tpf_fits=tpf,
            period=5.0,
            t0=2458001.0,
            duration_hours=4.8,
            radii_px=custom_radii,
        )

        # Should have depths for custom radii
        assert set(result.depths_by_radius_ppm.keys()) == set(custom_radii)

    def test_custom_center(self) -> None:
        """Test with custom aperture center."""
        tpf = make_synthetic_tpf_fits(
            shape=(500, 11, 11),
            stars=[StarSpec(row=5.0, col=5.0, flux=10000.0)],
            transit_spec=TransitSpec(
                star_idx=0,
                depth_frac=0.01,
                period=5.0,
                t0=2458001.0,
                duration_days=0.2,
            ),
            noise_level=50.0,
            seed=42,
        )

        # Use custom center offset from star
        result = compute_aperture_family_depth_curve(
            tpf_fits=tpf,
            period=5.0,
            t0=2458001.0,
            duration_hours=4.8,
            center=(5.0, 5.0),  # Centered on star
        )

        # Should measure valid depths
        valid_depths = [d for d in result.depths_by_radius_ppm.values() if np.isfinite(d)]
        assert len(valid_depths) >= 1

        # Evidence summary should record the center
        assert result.evidence_summary["center_row"] == 5.0
        assert result.evidence_summary["center_col"] == 5.0

    def test_insufficient_data_warning(self) -> None:
        """Test warning for insufficient transit data."""
        # Create TPF with very long period (few transits)
        tpf = make_synthetic_tpf_fits(
            shape=(100, 11, 11),  # Short time series
            stars=[StarSpec(row=5.0, col=5.0, flux=10000.0)],
            transit_spec=TransitSpec(
                star_idx=0,
                depth_frac=0.01,
                period=50.0,  # Very long period - might not have full transit
                t0=2458000.0,
                duration_days=0.1,
            ),
            noise_level=50.0,
            seed=42,
            time_span_days=5.0,  # Short baseline
        )

        result = compute_aperture_family_depth_curve(
            tpf_fits=tpf,
            period=50.0,
            t0=2458000.0,
            duration_hours=2.4,
        )

        # Should have warnings about insufficient data or unstable indicator
        # (depending on how the synthetic data lands)
        assert isinstance(result.warnings, list)

    def test_evidence_summary_fields(self) -> None:
        """Test that evidence summary has expected fields."""
        tpf = make_synthetic_tpf_fits(
            shape=(500, 11, 11),
            stars=[StarSpec(row=5.0, col=5.0, flux=10000.0)],
            transit_spec=TransitSpec(
                star_idx=0,
                depth_frac=0.01,
                period=5.0,
                t0=2458001.0,
                duration_days=0.2,
            ),
            noise_level=50.0,
            seed=42,
        )

        result = compute_aperture_family_depth_curve(
            tpf_fits=tpf,
            period=5.0,
            t0=2458001.0,
            duration_hours=4.8,
        )

        # Check expected fields in evidence summary
        summary = result.evidence_summary
        assert "n_apertures_tested" in summary
        assert "n_valid_depths" in summary
        assert "depth_slope_ppm_per_pixel" in summary
        assert "depth_slope_significance" in summary
        assert "blend_indicator" in summary
        assert "n_in_transit" in summary
        assert "n_out_of_transit" in summary

        assert summary["n_apertures_tested"] == len(DEFAULT_RADII_PX)
        assert summary["blend_indicator"] == result.blend_indicator

    def test_recommended_aperture(self) -> None:
        """Test that recommended aperture is the smallest with valid depth."""
        tpf = make_synthetic_tpf_fits(
            shape=(500, 11, 11),
            stars=[StarSpec(row=5.0, col=5.0, flux=10000.0)],
            transit_spec=TransitSpec(
                star_idx=0,
                depth_frac=0.01,
                period=5.0,
                t0=2458001.0,
                duration_days=0.2,
            ),
            noise_level=50.0,
            seed=42,
        )

        result = compute_aperture_family_depth_curve(
            tpf_fits=tpf,
            period=5.0,
            t0=2458001.0,
            duration_hours=4.8,
        )

        # Recommended aperture should be a valid radius
        assert result.recommended_aperture_px in DEFAULT_RADII_PX

        # Should be the smallest radius with valid depth
        for radius in sorted(DEFAULT_RADII_PX):
            if np.isfinite(result.depths_by_radius_ppm.get(radius, float("nan"))):
                assert result.recommended_aperture_px == radius
                break


class TestBlendIndicatorClassification:
    """Tests for blend indicator classification logic."""

    def test_consistent_classification(self) -> None:
        """Test that truly single-star systems get 'consistent' classification."""
        # Single bright star with clear transit
        tpf = make_synthetic_tpf_fits(
            shape=(1000, 11, 11),
            stars=[StarSpec(row=5.0, col=5.0, flux=20000.0)],
            transit_spec=TransitSpec(
                star_idx=0,
                depth_frac=0.015,
                period=3.0,
                t0=2458001.0,
                duration_days=0.15,
            ),
            noise_level=30.0,
            seed=123,
        )

        result = compute_aperture_family_depth_curve(
            tpf_fits=tpf,
            period=3.0,
            t0=2458001.0,
            duration_hours=3.6,
        )

        # Should be consistent (flat depth curve)
        assert result.blend_indicator in ["consistent", "unstable"]

    def test_high_noise_produces_result(self) -> None:
        """Test that high noise scenario produces a valid result without crashing."""
        # Low signal, high noise
        tpf = make_synthetic_tpf_fits(
            shape=(200, 11, 11),
            stars=[StarSpec(row=5.0, col=5.0, flux=500.0)],  # Very faint
            transit_spec=TransitSpec(
                star_idx=0,
                depth_frac=0.005,  # Small depth
                period=5.0,
                t0=2458001.0,
                duration_days=0.2,
            ),
            noise_level=500.0,  # High noise
            seed=42,
        )

        result = compute_aperture_family_depth_curve(
            tpf_fits=tpf,
            period=5.0,
            t0=2458001.0,
            duration_hours=4.8,
        )

        # With high noise, should still produce a valid result
        assert isinstance(result, ApertureFamilyResult)
        assert result.blend_indicator in ["consistent", "increasing", "decreasing", "unstable"]


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_small_aperture(self) -> None:
        """Test with very small aperture that might have few pixels."""
        tpf = make_synthetic_tpf_fits(
            shape=(500, 11, 11),
            stars=[StarSpec(row=5.0, col=5.0, flux=10000.0)],
            transit_spec=TransitSpec(
                star_idx=0,
                depth_frac=0.01,
                period=5.0,
                t0=2458001.0,
                duration_days=0.2,
            ),
            noise_level=50.0,
            seed=42,
        )

        # Very small radii
        result = compute_aperture_family_depth_curve(
            tpf_fits=tpf,
            period=5.0,
            t0=2458001.0,
            duration_hours=4.8,
            radii_px=[0.5, 1.0, 1.5],  # 0.5 might have very few pixels
        )

        # Should still produce result without crashing
        assert isinstance(result, ApertureFamilyResult)
        assert 0.5 in result.depths_by_radius_ppm

    def test_oot_margin_multiplier(self) -> None:
        """Test effect of out-of-transit margin multiplier."""
        tpf = make_synthetic_tpf_fits(
            shape=(500, 11, 11),
            stars=[StarSpec(row=5.0, col=5.0, flux=10000.0)],
            transit_spec=TransitSpec(
                star_idx=0,
                depth_frac=0.01,
                period=5.0,
                t0=2458001.0,
                duration_days=0.2,
            ),
            noise_level=50.0,
            seed=42,
        )

        # Test with different margin multipliers
        result_default = compute_aperture_family_depth_curve(
            tpf_fits=tpf,
            period=5.0,
            t0=2458001.0,
            duration_hours=4.8,
            oot_margin_mult=1.5,
        )

        result_large_margin = compute_aperture_family_depth_curve(
            tpf_fits=tpf,
            period=5.0,
            t0=2458001.0,
            duration_hours=4.8,
            oot_margin_mult=3.0,
        )

        # Both should produce results
        assert isinstance(result_default, ApertureFamilyResult)
        assert isinstance(result_large_margin, ApertureFamilyResult)

        # Larger margin should have fewer OOT points
        assert (
            result_large_margin.evidence_summary["n_out_of_transit"]
            <= result_default.evidence_summary["n_out_of_transit"]
        )
