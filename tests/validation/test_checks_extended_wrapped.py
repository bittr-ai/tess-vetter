from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from bittr_tess_vetter.domain.detection import TransitCandidate
from bittr_tess_vetter.domain.lightcurve import LightCurveData
from bittr_tess_vetter.validation.checks_extended_wrapped import (
    AliasDiagnosticsCheck,
    EphemerisReliabilityRegimeCheck,
    SectorConsistencyCheck,
)
from bittr_tess_vetter.validation.registry import CheckConfig, CheckInputs
from bittr_tess_vetter.validation.systematic_periods import compute_systematic_period_proximity


def _make_inputs(*, period_days: float = 3.0) -> CheckInputs:
    time = np.linspace(0.0, 27.4, 400, dtype=np.float64)
    flux = np.ones_like(time, dtype=np.float64)
    flux_err = np.full_like(time, 1e-4, dtype=np.float64)
    lc = LightCurveData(
        time=time,
        flux=flux,
        flux_err=flux_err,
        quality=np.zeros_like(time, dtype=np.int32),
        valid_mask=np.ones_like(time, dtype=bool),
        tic_id=123,
        sector=1,
        cadence_seconds=120,
    )
    candidate = TransitCandidate(
        period=float(period_days),
        t0=1.0,
        duration_hours=2.0,
        depth=0.001,
        snr=10.0,
    )
    return CheckInputs(lc=lc, candidate=candidate)


def test_compute_systematic_period_proximity_nominal() -> None:
    out = compute_systematic_period_proximity(period_days=13.0, threshold_fraction=0.05)
    assert out["nearest_systematic_period"] == 13.7
    assert out["systematic_period_name"] == "tess_orbital"
    assert out["nearest_systematic_days"] == 13.7
    assert np.isclose(out["fractional_distance"], 0.7 / 13.7)
    assert out["within_threshold"] is False
    assert out["threshold_fraction"] == 0.05


def test_v17_uniqueness_regime_labels(monkeypatch) -> None:
    check = EphemerisReliabilityRegimeCheck()
    inputs = _make_inputs()
    config = CheckConfig(extra_params={})

    def _fake_compute(**_kwargs):
        return SimpleNamespace(
            base=SimpleNamespace(score=2.0, depth_hat=1e-3, depth_sigma=2e-4),
            phase_shift_null=SimpleNamespace(p_value_one_sided=0.01, z_score=2.0),
            null_percentile=0.99,
            period_neighborhood=SimpleNamespace(
                best_period_days=3.0,
                best_score=2.0,
                second_best_score=1.85,
                peak_to_next=1.08,
                period_grid_days=np.array([2.99, 3.0, 3.01], dtype=np.float64),
                scores=np.array([1.8, 2.0, 1.85], dtype=np.float64),
            ),
            max_ablation_score_drop_fraction=0.1,
            top_contribution_fractions={"top_5_fraction_abs": 0.2},
            concentration=SimpleNamespace(n_in_transit=22, effective_n_points=380.0),
            to_dict=lambda: {},
        )

    monkeypatch.setattr(
        "bittr_tess_vetter.validation.ephemeris_reliability.compute_reliability_regime_numpy",
        _fake_compute,
    )
    monkeypatch.setattr(
        "bittr_tess_vetter.validation.ephemeris_specificity.phase_shift_t0s",
        lambda **_kwargs: np.array([1.1, 1.2], dtype=np.float64),
    )
    monkeypatch.setattr(
        "bittr_tess_vetter.validation.ephemeris_specificity.scores_for_t0s_numpy",
        lambda **_kwargs: np.array([1.6, 1.4], dtype=np.float64),
    )

    result = check.run(inputs, config)
    assert result.status == "ok"
    assert result.metrics["period_peak_to_next_ratio"] == 1.08
    assert result.metrics["uniqueness_regime"] == "marginal"
    assert "V17_REGIME_MARGINAL" in result.flags
    assert "Competing period has" in str(result.metrics["interpretation_note"])


def test_v19_includes_systematic_period_proximity(monkeypatch) -> None:
    check = AliasDiagnosticsCheck()
    inputs = _make_inputs(period_days=13.0)
    config = CheckConfig(extra_params={})

    monkeypatch.setattr(
        "bittr_tess_vetter.validation.alias_diagnostics.compute_harmonic_scores",
        lambda **_kwargs: [
            SimpleNamespace(harmonic="P", period=13.0, score=2.0, depth_ppm=200.0),
            SimpleNamespace(harmonic="P/2", period=6.5, score=1.2, depth_ppm=100.0),
        ],
    )
    monkeypatch.setattr(
        "bittr_tess_vetter.validation.alias_diagnostics.compute_secondary_significance",
        lambda **_kwargs: 0.3,
    )
    monkeypatch.setattr(
        "bittr_tess_vetter.validation.alias_diagnostics.detect_phase_shift_events",
        lambda **_kwargs: [SimpleNamespace(significance=1.4)],
    )

    result = check.run(inputs, config)
    assert result.status == "ok"
    assert result.metrics["nearest_systematic_period"] == 13.7
    assert result.metrics["systematic_period_name"] == "tess_orbital"
    assert np.isclose(float(result.metrics["fractional_distance"]), 0.7 / 13.7)
    assert result.metrics["systematic_nearest_period_days"] == 13.7
    assert result.metrics["systematic_within_threshold"] is False
    assert result.raw is not None
    assert "systematic_period_proximity" in result.raw


def test_v19_can_flag_near_systematic_harmonic() -> None:
    out = compute_systematic_period_proximity(period_days=6.85, threshold_fraction=0.05)
    assert out["within_threshold"] is True
    assert "tess_orbital_half" in str(out["systematic_period_name"])


def test_v21_reports_input_vs_used_sector_counts() -> None:
    check = SectorConsistencyCheck()
    inputs = _make_inputs()
    measurements = [
        {"sector": 1, "depth_ppm": 700.0, "depth_err_ppm": 50.0, "quality_weight": 1.0},
        {"sector": 2, "depth_ppm": 680.0, "depth_err_ppm": 55.0, "quality_weight": 1.0},
        {"sector": 3, "depth_ppm": 0.0, "depth_err_ppm": 1e12, "quality_weight": 0.0},
        {"sector": 4, "depth_ppm": -20.0, "depth_err_ppm": 100.0, "quality_weight": 1.0},
    ]
    inputs = CheckInputs(
        lc=inputs.lc,
        candidate=inputs.candidate,
        context={"sector_measurements": measurements},
    )
    result = check.run(inputs, CheckConfig(extra_params={}))
    assert result.status == "ok"
    assert result.metrics["n_sectors_input"] == 4
    assert result.metrics["n_sectors_used"] == 2
    assert result.metrics["n_sectors_excluded"] == 2
