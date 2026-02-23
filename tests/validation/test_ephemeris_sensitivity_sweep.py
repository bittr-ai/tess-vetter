from __future__ import annotations

import types

import numpy as np

import tess_vetter.validation.ephemeris_sensitivity_sweep as sweep
from tess_vetter.validation.ephemeris_sensitivity_sweep import (
    SweepRow,
    _apply_detrend,
    _apply_outlier_policy,
    _celerite2_sho_variant,
    _clean_finite_inputs,
    compute_sensitivity_sweep_numpy,
)
from tess_vetter.validation.ephemeris_specificity import SmoothTemplateConfig


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
        "tess_vetter.validation.ephemeris_sensitivity_sweep._score_variant",
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


def test_apply_outlier_policy_malformed_sigma_clip_falls_back_to_default_sigma() -> None:
    n = 20
    time = np.arange(n, dtype=np.float64)
    flux = np.ones(n, dtype=np.float64)
    flux[0] = 1_000.0
    flux_err = np.full(n, 1e-3, dtype=np.float64)

    out_t, out_f, out_fe, n_removed = _apply_outlier_policy(
        time,
        flux,
        flux_err,
        "sigma_clip_not_a_number",
    )

    assert n_removed == 1
    assert len(out_t) == n - 1
    assert len(out_f) == n - 1
    assert len(out_fe) == n - 1


def test_apply_outlier_policy_preserves_when_clip_would_remove_too_many_points() -> None:
    n = 20
    time = np.arange(n, dtype=np.float64)
    flux = np.array([0.0] * 10 + [1.0] * 10, dtype=np.float64)
    flux_err = np.full(n, 1e-3, dtype=np.float64)

    out_t, out_f, out_fe, n_removed = _apply_outlier_policy(time, flux, flux_err, "sigma_clip_0")

    assert n_removed == 0
    assert np.array_equal(out_t, time)
    assert np.array_equal(out_f, flux)
    assert np.array_equal(out_fe, flux_err)


def test_apply_detrend_unknown_name_returns_noop_meta() -> None:
    time = np.linspace(0.0, 2.0, 5, dtype=np.float64)
    flux = np.ones(5, dtype=np.float64)
    flux_err = np.full(5, 1e-4, dtype=np.float64)

    out_t, out_f, out_fe, meta = _apply_detrend(
        time,
        flux,
        flux_err,
        "totally_unknown_detrender",
        period_days=2.0,
        t0_btjd=0.5,
        duration_hours=2.0,
    )

    assert meta["detrend_name"] == "totally_unknown_detrender"
    assert meta["applied"] is False
    assert np.array_equal(out_t, time)
    assert np.array_equal(out_f, flux)
    assert np.array_equal(out_fe, flux_err)


def test_clean_finite_inputs_removes_nonfinite_rows() -> None:
    time = np.array([0.0, 1.0, np.nan, 3.0], dtype=np.float64)
    flux = np.array([1.0, np.inf, 1.0, 1.0], dtype=np.float64)
    flux_err = np.array([1e-3, 1e-3, 1e-3, np.nan], dtype=np.float64)

    out_t, out_f, out_fe, n_removed = _clean_finite_inputs(time, flux, flux_err)

    assert n_removed == 3
    assert np.array_equal(out_t, np.array([0.0], dtype=np.float64))
    assert np.array_equal(out_f, np.array([1.0], dtype=np.float64))
    assert np.array_equal(out_fe, np.array([1e-3], dtype=np.float64))


def test_compute_sensitivity_sweep_marks_failure_on_insufficient_points() -> None:
    n = 40
    time = np.linspace(0.0, 10.0, n, dtype=np.float64)
    flux = np.ones(n, dtype=np.float64)
    flux_err = np.full(n, 1e-4, dtype=np.float64)

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
    assert row.failure_reason == "Insufficient points after transforms (40 < 50)"
    assert any("Insufficient ok variants (0)" in note for note in result.notes)


def test_compute_sensitivity_sweep_captures_score_exception(monkeypatch) -> None:
    n = 80
    time = np.linspace(0.0, 10.0, n, dtype=np.float64)
    flux = np.ones(n, dtype=np.float64)
    flux_err = np.full(n, 1e-4, dtype=np.float64)

    def _boom(**_: object) -> tuple[float, float, float]:
        raise RuntimeError("intentional score failure")

    monkeypatch.setattr(
        "tess_vetter.validation.ephemeris_sensitivity_sweep._score_variant",
        _boom,
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

    row = result.sweep_table[0]
    assert row.status == "failed"
    assert row.failure_reason == "RuntimeError: intentional score failure"


def test_compute_sensitivity_sweep_adds_note_when_gp_variant_fails(monkeypatch) -> None:
    n = 80
    time = np.linspace(0.0, 10.0, n, dtype=np.float64)
    flux = np.ones(n, dtype=np.float64)
    flux_err = np.full(n, 1e-4, dtype=np.float64)

    def _fake_gp_variant(**_: object) -> SweepRow:
        return SweepRow(
            variant_id="celerite2_sho",
            status="failed",
            backend="cpu_gp",
            runtime_seconds=0.01,
            n_points_used=80,
            downsample_factor=None,
            outlier_policy=None,
            detrender="celerite2_sho",
            score=None,
            depth_hat_ppm=None,
            depth_err_ppm=None,
            warnings=[],
            failure_reason="synthetic gp failure",
            variant_config={"kernel": "sho+jitter"},
            gp_hyperparams=None,
            gp_fit_diagnostics=None,
        )

    monkeypatch.setattr(sweep, "_celerite2_sho_variant", _fake_gp_variant)

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
        include_celerite2_sho=True,
        random_seed=42,
    )

    assert result.n_variants_total == 2
    gp_rows = [row for row in result.sweep_table if row.variant_id == "celerite2_sho"]
    assert len(gp_rows) == 1
    assert gp_rows[0].status == "failed"
    assert any(note == "celerite2_sho variant failed: synthetic gp failure" for note in result.notes)


def _make_gp_test_arrays(n: int = 120) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = np.linspace(0.0, 20.0, n, dtype=np.float64)
    flux = np.ones(n, dtype=np.float64) + 1e-4 * np.sin(time)
    flux_err = np.full(n, 1e-4, dtype=np.float64)
    return time, flux, flux_err


class _DummySHOTerm:
    def __init__(self, *, sigma: float, rho: float, tau: float) -> None:
        self.sigma = sigma
        self.rho = rho
        self.tau = tau


class _DummyGP:
    def __init__(self, kernel: object, mean: float) -> None:
        self.kernel = kernel
        self.mean = mean
        self._n = 0

    def compute(self, time: np.ndarray, diag: np.ndarray) -> None:
        self._n = int(len(time))
        _ = diag

    def log_likelihood(self, flux: np.ndarray) -> float:
        return -float(np.sum((flux - 1.0) ** 2))

    def predict(self, flux: np.ndarray, t: np.ndarray, return_cov: bool = False) -> np.ndarray:
        _ = flux, return_cov
        return np.zeros_like(t, dtype=np.float64)


def test_celerite2_variant_success_path(monkeypatch) -> None:
    time, flux, flux_err = _make_gp_test_arrays()
    monkeypatch.setattr(sweep, "CELERITE2_AVAILABLE", True)
    monkeypatch.setattr(
        sweep,
        "celerite2_terms",
        types.SimpleNamespace(SHOTerm=_DummySHOTerm),
    )
    monkeypatch.setattr(
        sweep,
        "celerite2",
        types.SimpleNamespace(GaussianProcess=_DummyGP),
    )

    class _Result:
        success = True
        x = np.array([1e-3, 1.0, 1.0, 1e-4], dtype=np.float64)
        fun = 1.23
        nit = 5
        message = "ok"

    monkeypatch.setattr("scipy.optimize.minimize", lambda *args, **kwargs: _Result())
    monkeypatch.setattr(
        sweep,
        "_score_variant",
        lambda **kwargs: (12.0, 200.0, 30.0),
    )

    row = _celerite2_sho_variant(
        time=time,
        flux=flux,
        flux_err=flux_err,
        period_days=2.0,
        t0_btjd=0.5,
        duration_hours=2.0,
        config=SmoothTemplateConfig(),
        max_iterations=20,
        timeout_seconds=1.0,
        seed=1,
    )
    assert row.status == "ok"
    assert row.score == 12.0
    assert row.depth_hat_ppm == 200.0
    assert row.depth_err_ppm == 30.0
    assert row.gp_hyperparams is not None


def test_celerite2_variant_handles_fit_timeout(monkeypatch) -> None:
    time, flux, flux_err = _make_gp_test_arrays()
    monkeypatch.setattr(sweep, "CELERITE2_AVAILABLE", True)
    monkeypatch.setattr(
        sweep,
        "celerite2_terms",
        types.SimpleNamespace(SHOTerm=_DummySHOTerm),
    )
    monkeypatch.setattr(
        sweep,
        "celerite2",
        types.SimpleNamespace(GaussianProcess=_DummyGP),
    )

    class _StuckThread:
        def __init__(self, target, daemon: bool) -> None:
            self._target = target
            self._alive = True

        def start(self) -> None:
            return None

        def join(self, timeout: float | None = None) -> None:
            _ = timeout
            self._alive = True

        def is_alive(self) -> bool:
            return self._alive

    monkeypatch.setattr("threading.Thread", _StuckThread)

    row = _celerite2_sho_variant(
        time=time,
        flux=flux,
        flux_err=flux_err,
        period_days=2.0,
        t0_btjd=0.5,
        duration_hours=2.0,
        config=SmoothTemplateConfig(),
        max_iterations=10,
        timeout_seconds=0.01,
        seed=1,
    )
    assert row.status == "failed"
    assert row.failure_reason is not None
    assert "fit timeout" in row.failure_reason


def test_celerite2_variant_handles_fit_exception(monkeypatch) -> None:
    time, flux, flux_err = _make_gp_test_arrays()
    monkeypatch.setattr(sweep, "CELERITE2_AVAILABLE", True)
    monkeypatch.setattr(
        sweep,
        "celerite2_terms",
        types.SimpleNamespace(SHOTerm=_DummySHOTerm),
    )
    monkeypatch.setattr(
        sweep,
        "celerite2",
        types.SimpleNamespace(GaussianProcess=_DummyGP),
    )
    monkeypatch.setattr(
        "scipy.optimize.minimize",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("minimize boom")),
    )

    row = _celerite2_sho_variant(
        time=time,
        flux=flux,
        flux_err=flux_err,
        period_days=2.0,
        t0_btjd=0.5,
        duration_hours=2.0,
        config=SmoothTemplateConfig(),
        max_iterations=10,
        timeout_seconds=1.0,
        seed=1,
    )
    assert row.status == "failed"
    assert row.failure_reason is not None
    assert "fit exception:" in row.failure_reason


def test_celerite2_variant_handles_missing_fit_params(monkeypatch) -> None:
    time, flux, flux_err = _make_gp_test_arrays()
    monkeypatch.setattr(sweep, "CELERITE2_AVAILABLE", True)
    monkeypatch.setattr(
        sweep,
        "celerite2_terms",
        types.SimpleNamespace(SHOTerm=_DummySHOTerm),
    )
    monkeypatch.setattr(
        sweep,
        "celerite2",
        types.SimpleNamespace(GaussianProcess=_DummyGP),
    )

    class _NoRunThread:
        def __init__(self, target, daemon: bool) -> None:
            self._target = target
            self._alive = False

        def start(self) -> None:
            return None

        def join(self, timeout: float | None = None) -> None:
            _ = timeout
            self._alive = False

        def is_alive(self) -> bool:
            return self._alive

    monkeypatch.setattr("threading.Thread", _NoRunThread)

    row = _celerite2_sho_variant(
        time=time,
        flux=flux,
        flux_err=flux_err,
        period_days=2.0,
        t0_btjd=0.5,
        duration_hours=2.0,
        config=SmoothTemplateConfig(),
        max_iterations=10,
        timeout_seconds=0.2,
        seed=1,
    )
    assert row.status == "failed"
    assert row.failure_reason == "fit returned no parameters"


def test_celerite2_variant_handles_non_finite_score_outputs(monkeypatch) -> None:
    time, flux, flux_err = _make_gp_test_arrays()
    monkeypatch.setattr(sweep, "CELERITE2_AVAILABLE", True)
    monkeypatch.setattr(
        sweep,
        "celerite2_terms",
        types.SimpleNamespace(SHOTerm=_DummySHOTerm),
    )
    monkeypatch.setattr(
        sweep,
        "celerite2",
        types.SimpleNamespace(GaussianProcess=_DummyGP),
    )

    class _Result:
        success = True
        x = np.array([1e-3, 1.0, 1.0, 1e-4], dtype=np.float64)
        fun = 0.5
        nit = 3
        message = "ok"

    monkeypatch.setattr("scipy.optimize.minimize", lambda *args, **kwargs: _Result())
    monkeypatch.setattr(
        sweep,
        "_score_variant",
        lambda **kwargs: (float("nan"), 100.0, 20.0),
    )

    row = _celerite2_sho_variant(
        time=time,
        flux=flux,
        flux_err=flux_err,
        period_days=2.0,
        t0_btjd=0.5,
        duration_hours=2.0,
        config=SmoothTemplateConfig(),
        max_iterations=20,
        timeout_seconds=1.0,
        seed=1,
    )
    assert row.status == "failed"
    assert row.failure_reason is not None
    assert "non-finite score outputs" in row.failure_reason
