from __future__ import annotations

import numpy as np
import pytest

from tess_vetter.validation import ephemeris_reliability as er
from tess_vetter.validation.ephemeris_specificity import (
    ConcentrationMetrics,
    LocalT0SensitivityResult,
    PhaseShiftNullResult,
    SmoothTemplateConfig,
    SmoothTemplateScoreResult,
)


def _inject_box_transit(
    time: np.ndarray,
    flux: np.ndarray,
    *,
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    depth_frac: float,
) -> np.ndarray:
    duration_days = float(duration_hours) / 24.0
    half = duration_days / 2.0
    phase = (time - float(t0_btjd)) / float(period_days)
    phase = phase - np.floor(phase + 0.5)
    dt_days = np.abs(phase * float(period_days))
    in_transit = dt_days <= half
    out = flux.copy()
    out[in_transit] -= float(depth_frac)
    return out


def test_compute_reliability_regime_numpy_smoke() -> None:
    rng = np.random.default_rng(123)
    n = 4000
    time = np.linspace(0.0, 27.4, n, dtype=np.float64)
    flux = np.ones(n, dtype=np.float64) + rng.normal(0.0, 2e-4, size=n).astype(np.float64)
    flux_err = np.full(n, 2e-4, dtype=np.float64)

    period_days = 3.2
    t0_btjd = 1.1
    duration_hours = 2.4
    flux = _inject_box_transit(
        time,
        flux,
        period_days=period_days,
        t0_btjd=t0_btjd,
        duration_hours=duration_hours,
        depth_frac=8e-4,
    )

    res = er.compute_reliability_regime_numpy(
        time=time,
        flux=flux,
        flux_err=flux_err,
        period_days=period_days,
        t0_btjd=t0_btjd,
        duration_hours=duration_hours,
        config=SmoothTemplateConfig(ingress_egress_fraction=0.2, sharpness=30.0),
        n_phase_shifts=50,
        period_jitter_n=21,
        include_harmonics=True,
    )

    assert np.isfinite(res.base.score)
    assert res.period_neighborhood.period_grid_days.size == 21
    assert 0.0 <= res.phase_shift_null.p_value_one_sided <= 1.0
    assert isinstance(res.warnings, list)
    assert isinstance(res.label, str)


def test_period_grid_around_enforces_min_size_and_clips_bounds() -> None:
    with pytest.raises(ValueError, match="period_jitter_n must be >= 3"):
        er._period_grid_around(period_days=2.0, period_jitter_frac=0.01, n=2)

    low = er._period_grid_around(period_days=0.01, period_jitter_frac=0.90, n=4)
    assert low.size == 5  # even n is promoted to odd
    assert np.all(low >= 0.05)

    high = er._period_grid_around(period_days=2000.0, period_jitter_frac=0.10, n=3)
    assert np.all(high <= 1000.0)
    assert np.all(high == 1000.0)


def test_compute_reliability_regime_numpy_raises_when_period_jitter_n_too_small() -> None:
    time = np.linspace(0.0, 1.0, 5, dtype=np.float64)
    flux = np.ones_like(time)
    flux_err = np.full_like(time, 1e-3)

    with pytest.raises(ValueError, match="period_jitter_n must be >= 3"):
        er.compute_reliability_regime_numpy(
            time=time,
            flux=flux,
            flux_err=flux_err,
            period_days=2.0,
            t0_btjd=0.5,
            duration_hours=2.0,
            config=SmoothTemplateConfig(),
            period_jitter_n=2,
        )


def test_compute_reliability_regime_numpy_low_reliability_overrides_pipeline_sensitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    time = np.linspace(0.0, 1.0, 5, dtype=np.float64)
    flux = np.array([0.90, 1.00, 1.00, 1.00, 1.00], dtype=np.float64)
    flux_err = np.full_like(time, 1e-3)

    def _fake_score_fixed_period_numpy(**kwargs):
        n = int(kwargs["time"].size)
        period = float(kwargs["period_days"])
        if n < 5:
            score = 1.0
        else:
            score = {
                1.0: 2.0,
                1.8: 9.0,
                2.0: 10.0,
                2.2: 8.0,
                4.0: 1.5,
            }.get(round(period, 1), 0.5)
        return SmoothTemplateScoreResult(
            score=score,
            depth_hat=1e-3,
            depth_sigma=1e-4,
            template=np.linspace(1.0, 0.2, n, dtype=np.float64),
        )

    monkeypatch.setattr(er, "score_fixed_period_numpy", _fake_score_fixed_period_numpy)
    monkeypatch.setattr(
        er,
        "compute_phase_shift_null",
        lambda **_: PhaseShiftNullResult(
            n_trials=10,
            strategy="grid",
            null_mean=0.0,
            null_std=1.0,
            z_score=0.0,
            p_value_one_sided=0.20,
        ),
    )
    monkeypatch.setattr(
        er,
        "compute_concentration_metrics",
        lambda **_: ConcentrationMetrics(
            in_transit_contribution_abs=0.8,
            max_point_fraction_abs=0.7,
            top_5_fraction_abs=0.7,
            effective_n_points=2.0,
            n_in_transit=3,
        ),
    )
    monkeypatch.setattr(
        er,
        "compute_local_t0_sensitivity_numpy",
        lambda **_: LocalT0SensitivityResult(
            backend="numpy",
            t0_best_btjd=0.5,
            score_at_input=9.5,
            score_best=10.0,
            delta_score=0.5,
            curvature=1.0,
            fwhm_minutes=30.0,
            n_grid=11,
            half_span_minutes=60.0,
        ),
    )

    res = er.compute_reliability_regime_numpy(
        time=time,
        flux=flux,
        flux_err=flux_err,
        period_days=2.0,
        t0_btjd=0.5,
        duration_hours=2.0,
        config=SmoothTemplateConfig(),
        period_jitter_frac=0.1,
        period_jitter_n=3,
        include_harmonics=True,
        p_value_warn_threshold=0.01,
        peak_ratio_warn_threshold=1.5,
        ablation_score_drop_warn_threshold=0.5,
        top_contribution_warn_fraction=0.35,
    )

    assert res.label == "low_reliability"
    assert len(res.warnings) == 4
    assert res.max_ablation_score_drop_fraction > 0.5
    assert set(res.harmonics) == {"period_x0.5", "period_x2"}


def test_compute_reliability_regime_numpy_pipeline_sensitive_period_warning_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    time = np.linspace(0.0, 1.0, 5, dtype=np.float64)
    flux = np.array([0.95, 1.00, 1.00, 1.00, 1.00], dtype=np.float64)
    flux_err = np.full_like(time, 1e-3)

    def _fake_score_fixed_period_numpy(**kwargs):
        n = int(kwargs["time"].size)
        period = float(kwargs["period_days"])
        score = {
            1.8: 9.0,
            2.0: 10.0,
            2.2: 8.0,
        }.get(round(period, 1), 9.9 if n < 5 else 0.5)
        return SmoothTemplateScoreResult(
            score=score,
            depth_hat=1e-3,
            depth_sigma=1e-4,
            template=np.linspace(1.0, 0.2, n, dtype=np.float64),
        )

    monkeypatch.setattr(er, "score_fixed_period_numpy", _fake_score_fixed_period_numpy)
    monkeypatch.setattr(
        er,
        "compute_phase_shift_null",
        lambda **_: PhaseShiftNullResult(
            n_trials=10,
            strategy="grid",
            null_mean=0.0,
            null_std=1.0,
            z_score=0.0,
            p_value_one_sided=0.001,
        ),
    )
    monkeypatch.setattr(
        er,
        "compute_concentration_metrics",
        lambda **_: ConcentrationMetrics(
            in_transit_contribution_abs=0.2,
            max_point_fraction_abs=0.2,
            top_5_fraction_abs=0.2,
            effective_n_points=5.0,
            n_in_transit=3,
        ),
    )
    monkeypatch.setattr(
        er,
        "compute_local_t0_sensitivity_numpy",
        lambda **_: LocalT0SensitivityResult(
            backend="numpy",
            t0_best_btjd=0.5,
            score_at_input=9.9,
            score_best=10.0,
            delta_score=0.1,
            curvature=1.0,
            fwhm_minutes=30.0,
            n_grid=11,
            half_span_minutes=60.0,
        ),
    )

    res = er.compute_reliability_regime_numpy(
        time=time,
        flux=flux,
        flux_err=flux_err,
        period_days=2.0,
        t0_btjd=0.5,
        duration_hours=2.0,
        config=SmoothTemplateConfig(),
        period_jitter_frac=0.1,
        period_jitter_n=3,
        include_harmonics=False,
        p_value_warn_threshold=0.01,
        peak_ratio_warn_threshold=1.5,
        ablation_score_drop_warn_threshold=1.1,
        top_contribution_warn_fraction=1.1,
    )

    assert res.label == "pipeline_sensitive"
    assert len(res.warnings) == 1
    assert "Period neighborhood confusable" in res.warnings[0]
    assert res.harmonics == {}


def test_compute_schedulability_summary_handles_non_finite_inputs() -> None:
    result = er.EphemerisReliabilityRegimeResult(
        base=SmoothTemplateScoreResult(
            score=1.0,
            depth_hat=1e-3,
            depth_sigma=1e-4,
            template=np.ones(5, dtype=np.float64),
        ),
        phase_shift_null=PhaseShiftNullResult(
            n_trials=10,
            strategy="grid",
            null_mean=0.0,
            null_std=1.0,
            z_score=0.0,
            p_value_one_sided=0.5,
        ),
        null_percentile=float("nan"),
        period_neighborhood=er.PeriodNeighborhoodResult(
            period_grid_days=np.array([1.9, 2.0, 2.1], dtype=np.float64),
            scores=np.array([0.9, 1.0, 0.8], dtype=np.float64),
            best_period_days=2.0,
            best_score=1.0,
            score_at_input=1.0,
            second_best_score=0.9,
            peak_to_next=float("nan"),
        ),
        harmonics={},
        concentration=ConcentrationMetrics(
            in_transit_contribution_abs=0.2,
            max_point_fraction_abs=0.2,
            top_5_fraction_abs=float("inf"),
            effective_n_points=5.0,
            n_in_transit=3,
        ),
        top_contribution_fractions={},
        ablation=[],
        max_ablation_score_drop_fraction=float("nan"),
        t0_sensitivity=LocalT0SensitivityResult(
            backend="numpy",
            t0_best_btjd=0.5,
            score_at_input=1.0,
            score_best=float("nan"),
            delta_score=float("nan"),
            curvature=float("nan"),
            fwhm_minutes=float("nan"),
            n_grid=11,
            half_span_minutes=60.0,
        ),
        label="ok",
        warnings=[],
    )

    summary = er.compute_schedulability_summary_from_regime_result(result)
    assert summary.scalar == 0.0
    assert all(value == 0.0 for value in summary.components.values())

    as_dict = result.to_dict()
    assert "schedulability_summary" in as_dict
    assert as_dict["schedulability_summary"]["scalar"] == 0.0
