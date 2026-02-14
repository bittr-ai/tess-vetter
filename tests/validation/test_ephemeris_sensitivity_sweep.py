from __future__ import annotations

import numpy as np

from bittr_tess_vetter.validation.ephemeris_sensitivity_sweep import (
    compute_sensitivity_sweep_numpy,
)
from bittr_tess_vetter.validation.ephemeris_specificity import SmoothTemplateConfig


def test_sweep_drops_non_finite_points_before_scoring() -> None:
    n = 80
    time = np.linspace(0.0, 10.0, n, dtype=np.float64)
    flux = np.ones(n, dtype=np.float64)
    flux_err = np.full(n, 1e-4, dtype=np.float64)

    # Inject a few non-finite rows that should be removed before scoring.
    time[3] = np.nan
    flux[7] = np.inf
    flux_err[11] = np.nan

    result = compute_sensitivity_sweep_numpy(
        time=time,
        flux=flux,
        flux_err=flux_err,
        period_days=2.0,
        t0_btjd=0.5,
        duration_hours=2.0,
        config=SmoothTemplateConfig(),
        downsample_levels=[1],
        outlier_policies=["none"],
        detrenders=["none"],
        include_celerite2_sho=False,
        random_seed=42,
    )

    assert result.n_variants_total == 1
    row = result.sweep_table[0]
    assert row.status == "ok"
    assert row.n_points_used == n - 3
    assert any("non-finite points before scoring" in warning for warning in row.warnings)
    assert row.failure_reason is None
    assert row.score is not None and np.isfinite(row.score)
    assert row.depth_hat_ppm is not None and np.isfinite(row.depth_hat_ppm)
    assert row.depth_err_ppm is not None and np.isfinite(row.depth_err_ppm)


def test_sweep_marks_variant_failed_when_score_outputs_non_finite(monkeypatch) -> None:
    n = 80
    time = np.linspace(0.0, 10.0, n, dtype=np.float64)
    flux = np.ones(n, dtype=np.float64)
    flux_err = np.full(n, 1e-4, dtype=np.float64)

    def _fake_score_variant(**_: object) -> tuple[float, float, float]:
        return float("nan"), 123.0, 45.0

    monkeypatch.setattr(
        "bittr_tess_vetter.validation.ephemeris_sensitivity_sweep._score_variant",
        _fake_score_variant,
    )

    result = compute_sensitivity_sweep_numpy(
        time=time,
        flux=flux,
        flux_err=flux_err,
        period_days=2.0,
        t0_btjd=0.5,
        duration_hours=2.0,
        config=SmoothTemplateConfig(),
        downsample_levels=[1],
        outlier_policies=["none"],
        detrenders=["none"],
        include_celerite2_sho=False,
        random_seed=42,
    )

    assert result.n_variants_total == 1
    assert result.n_variants_ok == 0
    row = result.sweep_table[0]
    assert row.status == "failed"
    assert row.score is None
    assert row.depth_hat_ppm is None
    assert row.depth_err_ppm is None
    assert row.failure_reason is not None
    assert "non-finite score outputs" in row.failure_reason
