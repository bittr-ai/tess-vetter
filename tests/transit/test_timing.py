"""Tests for transit timing measurement primitives.

Tests for:
- measure_single_transit: Single transit fitting
- measure_all_transit_times: Full light curve transit measurement
- compute_ttv_statistics: TTV analysis
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from bittr_tess_vetter.transit import (
    TransitTime,
    TTVResult,
    compute_ttv_statistics,
    measure_all_transit_times,
    measure_single_transit,
)
from bittr_tess_vetter.transit.timing import measure_all_transit_times_with_diagnostics


class TestMeasureSingleTransit:
    """Tests for single transit fitting."""

    def test_recovers_transit_center(
        self, multi_transit_lc: dict[str, NDArray[np.float64] | float | int]
    ) -> None:
        """Correctly measures mid-transit time."""
        time = multi_transit_lc["time"]
        flux = multi_transit_lc["flux"]
        flux_err = multi_transit_lc["flux_err"]
        t0 = multi_transit_lc["t0"]
        duration_hours = multi_transit_lc["transit_duration_hours"]

        assert isinstance(time, np.ndarray)
        assert isinstance(flux, np.ndarray)
        assert isinstance(flux_err, np.ndarray)
        assert isinstance(t0, float)
        assert isinstance(duration_hours, float)

        # Measure first transit
        tc, tc_err, depth, dur, snr, converged = measure_single_transit(
            time, flux, flux_err, t0, duration_hours
        )

        assert converged
        assert abs(tc - t0) < 0.05  # Within ~1 hour of expected
        assert tc_err > 0
        assert tc_err < 0.1  # Reasonable uncertainty
        assert depth > 0.001  # Measurable depth
        assert snr > 2.0  # Significant detection

    def test_handles_missing_transit(self) -> None:
        """Returns unconverged for time without transit."""
        np.random.seed(42)
        time = np.linspace(0, 10, 1000, dtype=np.float64)
        flux = 1.0 + np.random.normal(0, 0.001, len(time))
        flux = flux.astype(np.float64)
        flux_err = np.ones_like(flux) * 0.001

        # Try to measure transit at time where there is none
        tc, tc_err, depth, dur, snr, converged = measure_single_transit(
            time, flux, flux_err, 5.0, 2.0
        )

        # Should have low SNR or small depth
        assert snr < 3.0 or depth < 0.001

    def test_recovers_depth(
        self, multi_transit_lc: dict[str, NDArray[np.float64] | float | int]
    ) -> None:
        """Correctly measures transit depth."""
        time = multi_transit_lc["time"]
        flux = multi_transit_lc["flux"]
        flux_err = multi_transit_lc["flux_err"]
        t0 = multi_transit_lc["t0"]
        transit_depth = multi_transit_lc["transit_depth"]
        duration_hours = multi_transit_lc["transit_duration_hours"]

        assert isinstance(time, np.ndarray)
        assert isinstance(flux, np.ndarray)
        assert isinstance(flux_err, np.ndarray)
        assert isinstance(t0, float)
        assert isinstance(transit_depth, float)
        assert isinstance(duration_hours, float)

        tc, tc_err, depth, dur, snr, converged = measure_single_transit(
            time, flux, flux_err, t0, duration_hours
        )

        assert converged
        # Depth should be within 50% of true value
        assert abs(depth - transit_depth) < 0.5 * transit_depth

    def test_handles_insufficient_data(self) -> None:
        """Returns unconverged for insufficient data."""
        time = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        flux = np.array([1.0, 0.99, 1.0], dtype=np.float64)
        flux_err = np.ones_like(flux) * 0.001

        tc, tc_err, depth, dur, snr, converged = measure_single_transit(
            time, flux, flux_err, 2.0, 2.0
        )

        assert not converged


class TestMeasureAllTransitTimes:
    """Tests for measuring all transits in a light curve."""

    def test_measures_multiple_transits(
        self, multi_transit_lc: dict[str, NDArray[np.float64] | float | int]
    ) -> None:
        """Measures all transits in the light curve."""
        time = multi_transit_lc["time"]
        flux = multi_transit_lc["flux"]
        flux_err = multi_transit_lc["flux_err"]
        period = multi_transit_lc["period"]
        t0 = multi_transit_lc["t0"]
        duration_hours = multi_transit_lc["transit_duration_hours"]

        assert isinstance(time, np.ndarray)
        assert isinstance(flux, np.ndarray)
        assert isinstance(flux_err, np.ndarray)
        assert isinstance(period, float)
        assert isinstance(t0, float)
        assert isinstance(duration_hours, float)

        transit_times = measure_all_transit_times(
            time, flux, flux_err, period, t0, duration_hours, min_snr=2.0
        )

        # Should find multiple transits
        assert len(transit_times) >= 5

        # All should be TransitTime objects
        for tt in transit_times:
            assert isinstance(tt, TransitTime)
            assert tt.snr >= 2.0

    def test_returns_empty_for_flat_lc(self) -> None:
        """Returns empty list for flat light curve."""
        np.random.seed(42)
        time = np.linspace(0, 100, 10000, dtype=np.float64)
        flux = 1.0 + np.random.normal(0, 0.001, len(time))
        flux = flux.astype(np.float64)
        flux_err = np.ones_like(flux) * 0.001

        transit_times = measure_all_transit_times(time, flux, flux_err, 10.0, 5.0, 3.0, min_snr=3.0)

        # Should find no significant transits
        assert len(transit_times) == 0

    def test_min_snr_threshold(
        self, multi_transit_lc: dict[str, NDArray[np.float64] | float | int]
    ) -> None:
        """Respects minimum SNR threshold."""
        time = multi_transit_lc["time"]
        flux = multi_transit_lc["flux"]
        flux_err = multi_transit_lc["flux_err"]
        period = multi_transit_lc["period"]
        t0 = multi_transit_lc["t0"]
        duration_hours = multi_transit_lc["transit_duration_hours"]

        assert isinstance(time, np.ndarray)
        assert isinstance(flux, np.ndarray)
        assert isinstance(flux_err, np.ndarray)
        assert isinstance(period, float)
        assert isinstance(t0, float)
        assert isinstance(duration_hours, float)

        # High SNR threshold
        transit_times_high = measure_all_transit_times(
            time, flux, flux_err, period, t0, duration_hours, min_snr=5.0
        )

        # Low SNR threshold
        transit_times_low = measure_all_transit_times(
            time, flux, flux_err, period, t0, duration_hours, min_snr=1.0
        )

        # Lower threshold should find more (or equal) transits
        assert len(transit_times_low) >= len(transit_times_high)

    def test_transit_time_fields(
        self, multi_transit_lc: dict[str, NDArray[np.float64] | float | int]
    ) -> None:
        """TransitTime objects have all required fields."""
        time = multi_transit_lc["time"]
        flux = multi_transit_lc["flux"]
        flux_err = multi_transit_lc["flux_err"]
        period = multi_transit_lc["period"]
        t0 = multi_transit_lc["t0"]
        duration_hours = multi_transit_lc["transit_duration_hours"]

        assert isinstance(time, np.ndarray)
        assert isinstance(flux, np.ndarray)
        assert isinstance(flux_err, np.ndarray)
        assert isinstance(period, float)
        assert isinstance(t0, float)
        assert isinstance(duration_hours, float)

        transit_times = measure_all_transit_times(
            time, flux, flux_err, period, t0, duration_hours, min_snr=2.0
        )

        if len(transit_times) > 0:
            tt = transit_times[0]
            assert isinstance(tt.epoch, int)
            assert isinstance(tt.tc, float)
            assert isinstance(tt.tc_err, float)
            assert isinstance(tt.depth_ppm, float)
            assert isinstance(tt.duration_hours, float)
            assert isinstance(tt.snr, float)

    def test_with_diagnostics_includes_reject_reasons(self) -> None:
        np.random.seed(7)
        time = np.linspace(0, 100, 4000, dtype=np.float64)
        flux = 1.0 + np.random.normal(0, 0.0015, len(time))
        flux = flux.astype(np.float64)
        flux_err = np.ones_like(flux) * 0.0015

        transit_times, diagnostics = measure_all_transit_times_with_diagnostics(
            time=time,
            flux=flux,
            flux_err=flux_err,
            period=10.0,
            t0=5.0,
            duration_hours=3.0,
            min_snr=3.0,
        )

        assert isinstance(transit_times, list)
        assert diagnostics["attempted_epochs"] >= diagnostics["accepted_epochs"]
        assert isinstance(diagnostics["reject_counts"], dict)
        assert isinstance(diagnostics["epoch_details"], list)
        if len(diagnostics["epoch_details"]) > 0:
            row = diagnostics["epoch_details"][0]
            assert "epoch" in row
            assert "accepted" in row
            assert "reject_reason" in row

    def test_adaptive_windows_recover_shifted_transit(self) -> None:
        np.random.seed(11)
        period = 8.0
        t0_catalog = 100.0
        t0_observed = t0_catalog + (4.0 / 24.0)  # 4-hour offset
        duration_hours = 2.0

        time = np.linspace(95.0, 130.0, 3500, dtype=np.float64)
        flux = 1.0 + np.random.normal(0.0, 0.0006, len(time))
        flux = flux.astype(np.float64)
        duration_days = duration_hours / 24.0
        in_transit = np.abs(((time - t0_observed + 0.5 * period) % period) - 0.5 * period) < (
            duration_days / 2.0
        )
        flux[in_transit] -= 0.003
        flux_err = np.ones_like(flux) * 0.0006

        transit_times, diagnostics = measure_all_transit_times_with_diagnostics(
            time=time,
            flux=flux,
            flux_err=flux_err,
            period=period,
            t0=t0_catalog,
            duration_hours=duration_hours,
            min_snr=2.0,
        )

        assert len(transit_times) >= 1
        assert diagnostics["accepted_epochs"] >= 1


class TestComputeTTVStatistics:
    """Tests for TTV statistics computation."""

    def test_computes_rms(
        self, multi_transit_lc: dict[str, NDArray[np.float64] | float | int]
    ) -> None:
        """Computes O-C RMS correctly."""
        time = multi_transit_lc["time"]
        flux = multi_transit_lc["flux"]
        flux_err = multi_transit_lc["flux_err"]
        period = multi_transit_lc["period"]
        t0 = multi_transit_lc["t0"]
        duration_hours = multi_transit_lc["transit_duration_hours"]

        assert isinstance(time, np.ndarray)
        assert isinstance(flux, np.ndarray)
        assert isinstance(flux_err, np.ndarray)
        assert isinstance(period, float)
        assert isinstance(t0, float)
        assert isinstance(duration_hours, float)

        transit_times = measure_all_transit_times(
            time, flux, flux_err, period, t0, duration_hours, min_snr=2.0
        )

        ttv_result = compute_ttv_statistics(transit_times, period, t0)

        assert isinstance(ttv_result, TTVResult)
        assert ttv_result.rms_seconds >= 0
        assert len(ttv_result.o_minus_c) == len(transit_times)

    def test_detects_ttv_signal(
        self, ttv_lc: dict[str, NDArray[np.float64] | float | list[float]]
    ) -> None:
        """Detects injected TTV signal."""
        time = ttv_lc["time"]
        flux = ttv_lc["flux"]
        flux_err = ttv_lc["flux_err"]
        period = ttv_lc["period"]
        t0 = ttv_lc["t0"]
        duration_hours = ttv_lc["transit_duration_hours"]
        ttv_amplitude = ttv_lc["ttv_amplitude_days"]

        assert isinstance(time, np.ndarray)
        assert isinstance(flux, np.ndarray)
        assert isinstance(flux_err, np.ndarray)
        assert isinstance(period, float)
        assert isinstance(t0, float)
        assert isinstance(duration_hours, float)
        assert isinstance(ttv_amplitude, float)

        transit_times = measure_all_transit_times(
            time, flux, flux_err, period, t0, duration_hours, min_snr=2.0
        )

        ttv_result = compute_ttv_statistics(transit_times, period, t0)

        # RMS should reflect the TTV amplitude
        # TTV amplitude is 0.02 days = 1728 seconds
        # RMS of sinusoid is amplitude / sqrt(2) ~ 1222 seconds
        expected_rms = ttv_amplitude * 86400 / np.sqrt(2)
        assert ttv_result.rms_seconds > expected_rms * 0.3  # Within factor of 3

    def test_empty_transit_list(self) -> None:
        """Handles empty transit list."""
        ttv_result = compute_ttv_statistics([], 10.0, 5.0)

        assert ttv_result.n_transits == 0
        assert ttv_result.rms_seconds == 0.0
        assert len(ttv_result.o_minus_c) == 0

    def test_to_dict_method(
        self, multi_transit_lc: dict[str, NDArray[np.float64] | float | int]
    ) -> None:
        """TTVResult.to_dict() returns valid dictionary."""
        time = multi_transit_lc["time"]
        flux = multi_transit_lc["flux"]
        flux_err = multi_transit_lc["flux_err"]
        period = multi_transit_lc["period"]
        t0 = multi_transit_lc["t0"]
        duration_hours = multi_transit_lc["transit_duration_hours"]

        assert isinstance(time, np.ndarray)
        assert isinstance(flux, np.ndarray)
        assert isinstance(flux_err, np.ndarray)
        assert isinstance(period, float)
        assert isinstance(t0, float)
        assert isinstance(duration_hours, float)

        transit_times = measure_all_transit_times(
            time, flux, flux_err, period, t0, duration_hours, min_snr=2.0
        )

        ttv_result = compute_ttv_statistics(transit_times, period, t0)
        result_dict = ttv_result.to_dict()

        assert "n_transits" in result_dict
        assert "rms_seconds" in result_dict
        assert "periodicity_score" in result_dict
        assert "periodicity_sigma" in result_dict
        assert "o_minus_c" in result_dict
        assert "transit_times" in result_dict


class TestIntegration:
    """Integration tests for full timing workflow."""

    def test_full_pipeline_no_ttv(
        self, multi_transit_lc: dict[str, NDArray[np.float64] | float | int]
    ) -> None:
        """Full pipeline on light curve without TTVs."""
        time = multi_transit_lc["time"]
        flux = multi_transit_lc["flux"]
        flux_err = multi_transit_lc["flux_err"]
        period = multi_transit_lc["period"]
        t0 = multi_transit_lc["t0"]
        duration_hours = multi_transit_lc["transit_duration_hours"]

        assert isinstance(time, np.ndarray)
        assert isinstance(flux, np.ndarray)
        assert isinstance(flux_err, np.ndarray)
        assert isinstance(period, float)
        assert isinstance(t0, float)
        assert isinstance(duration_hours, float)

        # Measure transits
        transit_times = measure_all_transit_times(
            time, flux, flux_err, period, t0, duration_hours, min_snr=2.0
        )

        # Compute statistics
        ttv_result = compute_ttv_statistics(transit_times, period, t0)

        # Should have low RMS (no TTVs)
        # Without TTVs, RMS should be dominated by measurement noise
        assert ttv_result.rms_seconds < 1000  # Less than ~17 minutes

    def test_full_pipeline_with_ttv(
        self, ttv_lc: dict[str, NDArray[np.float64] | float | list[float]]
    ) -> None:
        """Full pipeline on light curve with injected TTVs."""
        time = ttv_lc["time"]
        flux = ttv_lc["flux"]
        flux_err = ttv_lc["flux_err"]
        period = ttv_lc["period"]
        t0 = ttv_lc["t0"]
        duration_hours = ttv_lc["transit_duration_hours"]
        ttv_amplitude = ttv_lc["ttv_amplitude_days"]

        assert isinstance(time, np.ndarray)
        assert isinstance(flux, np.ndarray)
        assert isinstance(flux_err, np.ndarray)
        assert isinstance(period, float)
        assert isinstance(t0, float)
        assert isinstance(duration_hours, float)
        assert isinstance(ttv_amplitude, float)

        # Measure transits
        transit_times = measure_all_transit_times(
            time, flux, flux_err, period, t0, duration_hours, min_snr=2.0
        )

        # Should find many transits
        assert len(transit_times) >= 10

        # Compute statistics
        ttv_result = compute_ttv_statistics(transit_times, period, t0)

        # Should have substantial RMS reflecting TTV signal
        # TTV amplitude is 0.02 days = 1728 seconds
        assert ttv_result.rms_seconds > 500  # At least several minutes


class TestOutlierDetection:
    """Tests for outlier detection in transit timing."""

    def test_flags_oc_outliers(self) -> None:
        """Flags transits with large O-C deviations."""
        # Create transit times with one outlier (large O-C)
        normal_times = [
            TransitTime(
                epoch=i,
                tc=100.0 + i * 10.0,
                tc_err=0.01,
                depth_ppm=1000,
                duration_hours=2.0,
                snr=5.0,
            )
            for i in range(10)
        ]
        # Add one outlier with large timing offset
        outlier = TransitTime(
            epoch=10,
            tc=200.5,
            tc_err=0.01,  # 0.5 days off from expected 200.0
            depth_ppm=1000,
            duration_hours=2.0,
            snr=5.0,
        )
        all_times = normal_times + [outlier]

        ttv_result = compute_ttv_statistics(
            all_times, period=10.0, t0=100.0, expected_duration_hours=2.0
        )

        # The outlier should be flagged
        outlier_transits = [t for t in ttv_result.transit_times if t.is_outlier]
        assert len(outlier_transits) >= 1
        # Check that the epoch 10 transit is flagged
        epoch_10_transit = next((t for t in ttv_result.transit_times if t.epoch == 10), None)
        assert epoch_10_transit is not None
        assert epoch_10_transit.is_outlier
        assert epoch_10_transit.outlier_reason is not None
        assert "O-C" in epoch_10_transit.outlier_reason

    def test_flags_duration_outliers(self) -> None:
        """Flags transits with anomalous duration."""
        # Create transit times with one duration outlier
        normal_times = [
            TransitTime(
                epoch=i,
                tc=100.0 + i * 10.0,
                tc_err=0.01,
                depth_ppm=1000,
                duration_hours=2.0,
                snr=5.0,
            )
            for i in range(10)
        ]
        # Add one with very different duration (>50% different)
        duration_outlier = TransitTime(
            epoch=10,
            tc=200.0,
            tc_err=0.01,  # Normal timing
            depth_ppm=1000,
            duration_hours=4.5,
            snr=5.0,  # 125% longer than expected 2h
        )
        all_times = normal_times + [duration_outlier]

        ttv_result = compute_ttv_statistics(
            all_times, period=10.0, t0=100.0, expected_duration_hours=2.0
        )

        # The duration outlier should be flagged
        epoch_10_transit = next((t for t in ttv_result.transit_times if t.epoch == 10), None)
        assert epoch_10_transit is not None
        assert epoch_10_transit.is_outlier
        assert epoch_10_transit.outlier_reason is not None
        assert "duration" in epoch_10_transit.outlier_reason

    def test_normal_transits_not_flagged(
        self, multi_transit_lc: dict[str, NDArray[np.float64] | float | int]
    ) -> None:
        """Normal transits should not be flagged as outliers."""
        time = multi_transit_lc["time"]
        flux = multi_transit_lc["flux"]
        flux_err = multi_transit_lc["flux_err"]
        period = multi_transit_lc["period"]
        t0 = multi_transit_lc["t0"]
        duration_hours = multi_transit_lc["transit_duration_hours"]

        assert isinstance(time, np.ndarray)
        assert isinstance(flux, np.ndarray)
        assert isinstance(flux_err, np.ndarray)
        assert isinstance(period, float)
        assert isinstance(t0, float)
        assert isinstance(duration_hours, float)

        transit_times = measure_all_transit_times(
            time, flux, flux_err, period, t0, duration_hours, min_snr=2.0
        )

        ttv_result = compute_ttv_statistics(
            transit_times, period, t0, expected_duration_hours=duration_hours
        )

        # Most transits should not be outliers in clean data
        # Edge transits may have partial coverage causing duration differences
        n_outliers = sum(1 for t in ttv_result.transit_times if t.is_outlier)
        n_total = len(ttv_result.transit_times)
        outlier_fraction = n_outliers / n_total if n_total > 0 else 0
        assert outlier_fraction < 0.3  # Less than 30% should be outliers

    def test_to_dict_includes_outlier_fields(self) -> None:
        """TTVResult.to_dict() includes outlier fields."""
        transit_times = [
            TransitTime(
                epoch=0,
                tc=100.0,
                tc_err=0.01,
                depth_ppm=1000,
                duration_hours=2.0,
                snr=5.0,
                is_outlier=True,
                outlier_reason="test reason",
            ),
            TransitTime(
                epoch=1,
                tc=110.0,
                tc_err=0.01,
                depth_ppm=1000,
                duration_hours=2.0,
                snr=5.0,
                is_outlier=False,
                outlier_reason=None,
            ),
        ]

        ttv_result = compute_ttv_statistics(transit_times, period=10.0, t0=100.0)
        result_dict = ttv_result.to_dict()

        # Check that transit_times in dict have outlier fields
        assert "transit_times" in result_dict
        assert len(result_dict["transit_times"]) == 2

        first_transit = result_dict["transit_times"][0]
        assert "is_outlier" in first_transit
        assert "outlier_reason" in first_transit
