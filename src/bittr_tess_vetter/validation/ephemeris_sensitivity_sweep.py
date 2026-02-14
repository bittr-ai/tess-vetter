from __future__ import annotations

import time as time_module
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from bittr_tess_vetter.api.detrend import bin_median_trend, sigma_clip
from bittr_tess_vetter.api.transit_masks import get_out_of_transit_mask
from bittr_tess_vetter.validation.ephemeris_specificity import (
    SmoothTemplateConfig,
    score_fixed_period_numpy,
)
from bittr_tess_vetter.validation.detrend_grid_defaults import (
    DEFAULT_TRANSIT_MASKED_BIN_HOURS,
    DEFAULT_TRANSIT_MASKED_BUFFER_FACTORS,
    DEFAULT_TRANSIT_MASKED_SIGMA_CLIPS,
)

# Optional dependency for "deep" preset. Kept optional on purpose.
try:
    import celerite2
    from celerite2 import terms as celerite2_terms

    CELERITE2_AVAILABLE = True
except Exception:
    celerite2 = None  # type: ignore[assignment]
    celerite2_terms = None  # type: ignore[assignment]
    CELERITE2_AVAILABLE = False


@dataclass(frozen=True)
class SweepVariant:
    variant_id: str
    downsample_factor: int | None
    outlier_policy: str | None
    detrender: str | None
    detrender_bin_hours: float | None = None
    detrender_buffer_factor: float | None = None
    detrender_sigma_clip: float | None = None
    is_gp: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "downsample_factor": self.downsample_factor,
            "outlier_policy": self.outlier_policy,
            "detrender": self.detrender,
            "detrender_bin_hours": self.detrender_bin_hours,
            "detrender_buffer_factor": self.detrender_buffer_factor,
            "detrender_sigma_clip": self.detrender_sigma_clip,
            "is_gp": bool(self.is_gp),
        }


@dataclass(frozen=True)
class SweepRow:
    variant_id: str
    status: str
    backend: str
    runtime_seconds: float
    n_points_used: int
    downsample_factor: int | None
    outlier_policy: str | None
    detrender: str | None
    score: float | None
    depth_hat_ppm: float | None
    depth_err_ppm: float | None
    warnings: list[str]
    failure_reason: str | None
    variant_config: dict[str, Any]
    gp_hyperparams: dict[str, Any] | None
    gp_fit_diagnostics: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "status": self.status,
            "backend": self.backend,
            "runtime_seconds": float(self.runtime_seconds),
            "n_points_used": int(self.n_points_used),
            "downsample_factor": self.downsample_factor,
            "outlier_policy": self.outlier_policy,
            "detrender": self.detrender,
            "score": self.score,
            "depth_hat_ppm": self.depth_hat_ppm,
            "depth_err_ppm": self.depth_err_ppm,
            "warnings": list(self.warnings),
            "failure_reason": self.failure_reason,
            "variant_config": dict(self.variant_config),
            "gp_hyperparams": self.gp_hyperparams,
            "gp_fit_diagnostics": self.gp_fit_diagnostics,
        }


@dataclass(frozen=True)
class SensitivitySweepResult:
    stable: bool
    metric_variance: float | None
    score_spread_iqr_over_median: float | None
    depth_spread_iqr_over_median: float | None
    n_variants_total: int
    n_variants_ok: int
    n_variants_failed: int
    best_variant_id: str | None
    worst_variant_id: str | None
    stability_threshold: float
    notes: list[str]
    sweep_table: list[SweepRow]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stable": bool(self.stable),
            "metric_variance": self.metric_variance,
            "score_spread_iqr_over_median": self.score_spread_iqr_over_median,
            "depth_spread_iqr_over_median": self.depth_spread_iqr_over_median,
            "n_variants_total": int(self.n_variants_total),
            "n_variants_ok": int(self.n_variants_ok),
            "n_variants_failed": int(self.n_variants_failed),
            "best_variant_id": self.best_variant_id,
            "worst_variant_id": self.worst_variant_id,
            "stability_threshold": float(self.stability_threshold),
            "notes": list(self.notes),
            "sweep_table": [row.to_dict() for row in self.sweep_table],
        }


def _downsample(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    flux_err: NDArray[np.float64],
    factor: int,
    seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    if factor <= 1:
        return time, flux, flux_err
    rng = np.random.default_rng(int(seed))
    offset = int(rng.integers(0, int(factor)))
    idx = np.arange(offset, len(time), int(factor))
    return time[idx], flux[idx], flux_err[idx]


def _apply_outlier_policy(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    flux_err: NDArray[np.float64],
    policy: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], int]:
    if policy == "none":
        return time, flux, flux_err, 0

    if policy.startswith("sigma_clip_"):
        try:
            sigma_thresh = float(policy.split("_")[-1])
        except Exception:
            sigma_thresh = 4.0

        median_flux = float(np.median(flux))
        mad = float(np.median(np.abs(flux - median_flux)))
        sigma_est = mad * 1.4826 if mad > 0 else float(np.std(flux))
        mask = np.abs(flux - median_flux) <= float(sigma_thresh) * float(sigma_est)
        n_removed = int(np.sum(~mask))
        if n_removed >= len(flux) - 10:
            return time, flux, flux_err, 0
        return time[mask], flux[mask], flux_err[mask], n_removed

    return time, flux, flux_err, 0


def _apply_running_median_detrend(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    flux_err: NDArray[np.float64],
    *,
    window_days: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], dict[str, Any]]:
    meta: dict[str, Any] = {"detrend_name": f"running_median_{window_days:g}d", "applied": True}

    if len(time) < 3:
        return time, flux, flux_err, meta

    cadence_days = float(np.median(np.diff(time)))
    window_cadences = max(3, int(float(window_days) / cadence_days))
    if window_cadences % 2 == 0:
        window_cadences += 1
    meta["window_days"] = float(window_days)
    meta["window_cadences"] = int(window_cadences)

    half_window = int(window_cadences) // 2
    n = int(len(flux))
    trend = np.zeros_like(flux)
    for i in range(n):
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)
        trend[i] = np.median(flux[start:end])

    median_trend = float(np.median(trend))
    flux_detrended = flux / trend * median_trend if median_trend > 0 else flux - trend + 1.0
    return time, flux_detrended.astype(np.float64), flux_err, meta


def _apply_transit_masked_bin_median_detrend(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    flux_err: NDArray[np.float64],
    *,
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    bin_hours: float,
    buffer_factor: float,
    sigma_clip_value: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], dict[str, Any]]:
    meta: dict[str, Any] = {
        "detrend_name": "transit_masked_bin_median",
        "applied": True,
        "bin_hours": float(bin_hours),
        "buffer_factor": float(buffer_factor),
        "sigma_clip": float(sigma_clip_value),
    }
    if len(time) < 3:
        return time, flux, flux_err, meta

    finite_mask = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err)
    oot_mask = get_out_of_transit_mask(
        np.asarray(time, dtype=np.float64),
        float(period_days),
        float(t0_btjd),
        float(duration_hours),
        buffer_factor=float(buffer_factor),
    )
    trend_fit_mask = finite_mask & oot_mask

    n_sigma_clipped = 0
    if int(np.sum(trend_fit_mask)) >= 3:
        clip_keep = sigma_clip(flux[trend_fit_mask], sigma=float(sigma_clip_value))
        n_sigma_clipped = int(np.sum(trend_fit_mask)) - int(np.sum(clip_keep))
        trend_indices = np.flatnonzero(trend_fit_mask)
        trend_fit_mask = trend_fit_mask.copy()
        trend_fit_mask[trend_indices[~clip_keep]] = False

    fit_flux = np.asarray(flux, dtype=np.float64).copy()
    fit_flux[~trend_fit_mask] = np.nan
    trend = bin_median_trend(
        np.asarray(time, dtype=np.float64),
        fit_flux,
        bin_hours=float(bin_hours),
        min_bin_points=1,
    )

    trend_ref = float(np.nanmedian(trend[trend_fit_mask])) if np.any(trend_fit_mask) else float(np.nanmedian(trend))
    if not np.isfinite(trend_ref) or trend_ref == 0.0:
        trend_ref = 1.0
    safe_trend = np.where(np.isfinite(trend) & (trend != 0.0), trend, trend_ref)

    detrended_flux = np.asarray(flux, dtype=np.float64) / safe_trend * trend_ref
    detrended_flux_err = np.asarray(flux_err, dtype=np.float64) / safe_trend * trend_ref
    meta["n_points"] = int(len(time))
    meta["n_trend_fit_points"] = int(np.sum(trend_fit_mask))
    meta["n_sigma_clipped"] = int(n_sigma_clipped)
    return (
        np.asarray(time, dtype=np.float64),
        detrended_flux.astype(np.float64),
        detrended_flux_err.astype(np.float64),
        meta,
    )


def _apply_detrend(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    flux_err: NDArray[np.float64],
    detrend_name: str,
    *,
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    bin_hours: float | None = None,
    buffer_factor: float | None = None,
    sigma_clip_value: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], dict[str, Any]]:
    meta: dict[str, Any] = {"detrend_name": detrend_name}
    if detrend_name == "none":
        meta["applied"] = False
        return time, flux, flux_err, meta
    if detrend_name.startswith("running_median_"):
        try:
            window_str = detrend_name.replace("running_median_", "").rstrip("d")
            window_days = float(window_str)
        except Exception:
            window_days = 0.5
        return _apply_running_median_detrend(time, flux, flux_err, window_days=float(window_days))
    if detrend_name == "transit_masked_bin_median":
        return _apply_transit_masked_bin_median_detrend(
            time,
            flux,
            flux_err,
            period_days=float(period_days),
            t0_btjd=float(t0_btjd),
            duration_hours=float(duration_hours),
            bin_hours=float(6.0 if bin_hours is None else bin_hours),
            buffer_factor=float(2.0 if buffer_factor is None else buffer_factor),
            sigma_clip_value=float(5.0 if sigma_clip_value is None else sigma_clip_value),
        )
    meta["applied"] = False
    return time, flux, flux_err, meta


def _score_variant(
    *,
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    flux_err: NDArray[np.float64],
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    config: SmoothTemplateConfig,
) -> tuple[float, float, float]:
    res = score_fixed_period_numpy(
        time=time,
        flux=flux,
        flux_err=flux_err,
        period_days=float(period_days),
        t0_btjd=float(t0_btjd),
        duration_hours=float(duration_hours),
        config=config,
    )
    return float(res.score), float(res.depth_hat * 1e6), float(res.depth_sigma * 1e6)


def _clean_finite_inputs(
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    flux_err: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], int]:
    finite_mask = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err)
    n_removed = int(len(time) - int(np.sum(finite_mask)))
    if n_removed <= 0:
        return time, flux, flux_err, 0
    return time[finite_mask], flux[finite_mask], flux_err[finite_mask], n_removed


def _score_outputs_are_finite(*, score: float, depth_hat_ppm: float, depth_err_ppm: float) -> bool:
    return bool(np.isfinite(score) and np.isfinite(depth_hat_ppm) and np.isfinite(depth_err_ppm))


def _celerite2_sho_variant(
    *,
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    flux_err: NDArray[np.float64],
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    config: SmoothTemplateConfig,
    max_iterations: int,
    timeout_seconds: float,
    seed: int,
) -> SweepRow:
    start_time = time_module.perf_counter()
    base: dict[str, Any] = {
        "variant_id": "celerite2_sho",
        "status": "failed",
        "backend": "cpu_gp",
        "runtime_seconds": 0.0,
        "n_points_used": int(len(time)),
        "downsample_factor": None,
        "outlier_policy": None,
        "detrender": "celerite2_sho",
        "score": None,
        "depth_hat_ppm": None,
        "depth_err_ppm": None,
        "warnings": [],
        "failure_reason": None,
        "variant_config": {
            "kernel": "sho+jitter",
            "max_iterations": int(max_iterations),
            "timeout_seconds": float(timeout_seconds),
            "seed": int(seed),
        },
        "gp_hyperparams": None,
        "gp_fit_diagnostics": None,
    }

    def _finalize(row: dict[str, Any]) -> SweepRow:
        row["runtime_seconds"] = float(time_module.perf_counter() - start_time)
        return SweepRow(**row)  # type: ignore[arg-type]

    if not CELERITE2_AVAILABLE:
        base["failure_reason"] = "celerite2 not installed"
        return _finalize(base)

    time, flux, flux_err, n_removed_non_finite = _clean_finite_inputs(time, flux, flux_err)
    base["n_points_used"] = int(len(time))
    if n_removed_non_finite > 0:
        base["warnings"].append(
            f"Removed {n_removed_non_finite} non-finite points before GP fitting/scoring"
        )

    if len(time) < 50:
        base["failure_reason"] = f"insufficient data points ({len(time)} < 50)"
        return _finalize(base)

    try:
        import threading

        from scipy.optimize import minimize

        flux_std = float(np.std(flux))
        median_err = float(np.median(flux_err))

        sigma_min = 0.0001 * flux_std
        sigma_max = 0.1 * flux_std
        sigma_init = 0.01 * flux_std

        rho_min, rho_max, rho_init = 0.1, 10.0, 1.0
        tau_min, tau_max, tau_init = 0.1, 10.0, 1.0

        jitter_min = 0.5 * median_err
        jitter_max = 2.0 * median_err
        jitter_init = median_err

        kernel = celerite2_terms.SHOTerm(sigma=sigma_init, rho=rho_init, tau=tau_init)
        gp = celerite2.GaussianProcess(kernel, mean=1.0)
        diag = flux_err**2 + jitter_init**2
        gp.compute(time, diag=diag)

        def neg_log_likelihood(params: NDArray[np.float64]) -> float:
            sigma_val, rho_val, tau_val, jitter_val = params
            try:
                gp.kernel = celerite2_terms.SHOTerm(sigma=sigma_val, rho=rho_val, tau=tau_val)
                gp.compute(time, diag=flux_err**2 + jitter_val**2)
                return -float(gp.log_likelihood(flux))
            except Exception:
                return 1e10

        x0 = np.array([sigma_init, rho_init, tau_init, jitter_init], dtype=np.float64)
        bounds = [
            (sigma_min, sigma_max),
            (rho_min, rho_max),
            (tau_min, tau_max),
            (jitter_min, jitter_max),
        ]

        fit_result: dict[str, Any] = {
            "converged": False,
            "params": None,
            "loss_value": None,
            "n_iterations": 0,
            "message": "",
        }
        fit_exc: list[Exception] = []
        timeout_event = threading.Event()

        def fit_worker() -> None:
            nonlocal fit_result
            try:
                result = minimize(
                    neg_log_likelihood,
                    x0,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": int(max_iterations), "disp": False},
                )
                if not timeout_event.is_set():
                    fit_result = {
                        "converged": bool(result.success),
                        "params": result.x,
                        "loss_value": float(result.fun),
                        "n_iterations": int(getattr(result, "nit", 0)),
                        "message": str(getattr(result, "message", "")),
                    }
            except Exception as e:
                if not timeout_event.is_set():
                    fit_exc.append(e)

        thread = threading.Thread(target=fit_worker, daemon=True)
        thread.start()
        thread.join(timeout=float(timeout_seconds))

        if thread.is_alive():
            timeout_event.set()
            base["failure_reason"] = f"fit timeout after {timeout_seconds}s"
            return _finalize(base)
        if fit_exc:
            base["failure_reason"] = f"fit exception: {fit_exc[0]}"
            return _finalize(base)
        if fit_result["params"] is None:
            base["failure_reason"] = "fit returned no parameters"
            return _finalize(base)

        sigma_fit, rho_fit, tau_fit, jitter_fit = fit_result["params"]
        base["gp_hyperparams"] = {
            "sigma": float(sigma_fit),
            "rho": float(rho_fit),
            "tau": float(tau_fit),
            "jitter": float(jitter_fit),
        }
        notes: list[str] = []
        if not fit_result["converged"]:
            notes.append(f"optimizer did not converge: {fit_result['message']}")
            base["warnings"].append("GP fit did not converge; using best params found")
        base["gp_fit_diagnostics"] = {
            "converged": bool(fit_result["converged"]),
            "loss_value": fit_result["loss_value"],
            "n_iterations": int(fit_result["n_iterations"]),
            "notes": notes,
        }

        gp.kernel = celerite2_terms.SHOTerm(sigma=sigma_fit, rho=rho_fit, tau=tau_fit)
        gp.compute(time, diag=flux_err**2 + float(jitter_fit) ** 2)
        gp_mean = gp.predict(flux, t=time, return_cov=False)
        flux_detrended = flux - gp_mean + 1.0

        score, depth_hat_ppm, depth_err_ppm = _score_variant(
            time=time,
            flux=flux_detrended.astype(np.float64),
            flux_err=flux_err,
            period_days=period_days,
            t0_btjd=t0_btjd,
            duration_hours=duration_hours,
            config=config,
        )
        if not _score_outputs_are_finite(
            score=float(score),
            depth_hat_ppm=float(depth_hat_ppm),
            depth_err_ppm=float(depth_err_ppm),
        ):
            base["failure_reason"] = (
                "non-finite score outputs "
                f"(score={score}, depth_hat_ppm={depth_hat_ppm}, depth_err_ppm={depth_err_ppm})"
            )
            return _finalize(base)
        base["status"] = "ok"
        base["score"] = float(score)
        base["depth_hat_ppm"] = float(depth_hat_ppm)
        base["depth_err_ppm"] = float(depth_err_ppm)
        return _finalize(base)
    except Exception as e:
        base["failure_reason"] = f"{type(e).__name__}: {e}"
        return _finalize(base)


def compute_sensitivity_sweep_numpy(
    *,
    time: NDArray[np.float64],
    flux: NDArray[np.float64],
    flux_err: NDArray[np.float64],
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    config: SmoothTemplateConfig,
    downsample_levels: list[int] | None = None,
    outlier_policies: list[str] | None = None,
    detrenders: list[str] | None = None,
    include_celerite2_sho: bool = False,
    stability_threshold: float = 0.20,
    random_seed: int = 0,
    gp_max_iterations: int = 100,
    gp_timeout_seconds: float = 30.0,
) -> SensitivitySweepResult:
    downsample_levels = list(downsample_levels or [1, 2, 5])
    outlier_policies = list(outlier_policies or ["none", "sigma_clip_4"])
    detrenders = list(detrenders or ["none", "running_median_0.5d"])

    variants: list[SweepVariant] = []
    for ds in downsample_levels:
        for ol in outlier_policies:
            for dt in detrenders:
                if str(dt) == "transit_masked_bin_median":
                    for bin_hours in DEFAULT_TRANSIT_MASKED_BIN_HOURS:
                        for buffer_factor in DEFAULT_TRANSIT_MASKED_BUFFER_FACTORS:
                            for sigma_clip_value in DEFAULT_TRANSIT_MASKED_SIGMA_CLIPS:
                                variants.append(
                                    SweepVariant(
                                        variant_id=(
                                            f"ds{int(ds)}|ol_{ol}|dt_{dt}"
                                            f"|bh_{bin_hours:g}|bf_{buffer_factor:g}|sc_{sigma_clip_value:g}"
                                        ),
                                        downsample_factor=int(ds),
                                        outlier_policy=str(ol),
                                        detrender=str(dt),
                                        detrender_bin_hours=float(bin_hours),
                                        detrender_buffer_factor=float(buffer_factor),
                                        detrender_sigma_clip=float(sigma_clip_value),
                                        is_gp=False,
                                    )
                                )
                else:
                    variants.append(
                        SweepVariant(
                            variant_id=f"ds{int(ds)}|ol_{ol}|dt_{dt}",
                            downsample_factor=int(ds),
                            outlier_policy=str(ol),
                            detrender=str(dt),
                            is_gp=False,
                        )
                    )
    if include_celerite2_sho:
        variants.append(
            SweepVariant(
                variant_id="celerite2_sho",
                downsample_factor=None,
                outlier_policy=None,
                detrender=None,
                is_gp=True,
            )
        )

    notes: list[str] = []
    sweep_table: list[SweepRow] = []

    for variant in variants:
        start = time_module.time()
        base_row: dict[str, Any] = {
            "variant_id": variant.variant_id,
            "status": "ok",
            "backend": "numpy",
            "runtime_seconds": 0.0,
            "n_points_used": 0,
            "downsample_factor": variant.downsample_factor,
            "outlier_policy": variant.outlier_policy,
            "detrender": variant.detrender,
            "score": None,
            "depth_hat_ppm": None,
            "depth_err_ppm": None,
            "warnings": [],
            "failure_reason": None,
            "variant_config": variant.to_dict(),
            "gp_hyperparams": None,
            "gp_fit_diagnostics": None,
        }

        try:
            if variant.is_gp:
                row = _celerite2_sho_variant(
                    time=time,
                    flux=flux,
                    flux_err=flux_err,
                    period_days=float(period_days),
                    t0_btjd=float(t0_btjd),
                    duration_hours=float(duration_hours),
                    config=config,
                    max_iterations=int(gp_max_iterations),
                    timeout_seconds=float(gp_timeout_seconds),
                    seed=int(random_seed),
                )
                if row.status == "failed":
                    notes.append(f"celerite2_sho variant failed: {row.failure_reason}")
                sweep_table.append(
                    SweepRow(
                        **{**row.to_dict(), "runtime_seconds": float(time_module.time() - start)}
                    )
                )
                continue

            t_var, f_var, fe_var = time.copy(), flux.copy(), flux_err.copy()
            if variant.downsample_factor and variant.downsample_factor > 1:
                t_var, f_var, fe_var = _downsample(
                    t_var, f_var, fe_var, int(variant.downsample_factor), int(random_seed)
                )
            if variant.outlier_policy:
                t_var, f_var, fe_var, n_removed = _apply_outlier_policy(
                    t_var, f_var, fe_var, str(variant.outlier_policy)
                )
                if n_removed > 0:
                    base_row["warnings"].append(f"Removed {n_removed} outliers")
            if variant.detrender:
                t_var, f_var, fe_var, meta = _apply_detrend(
                    t_var,
                    f_var,
                    fe_var,
                    str(variant.detrender),
                    period_days=float(period_days),
                    t0_btjd=float(t0_btjd),
                    duration_hours=float(duration_hours),
                    bin_hours=variant.detrender_bin_hours,
                    buffer_factor=variant.detrender_buffer_factor,
                    sigma_clip_value=variant.detrender_sigma_clip,
                )
                if meta.get("applied"):
                    if str(variant.detrender) == "transit_masked_bin_median":
                        base_row["warnings"].append(
                            "Applied transit_masked_bin_median "
                            f"(bin_hours={meta.get('bin_hours')}, "
                            f"buffer_factor={meta.get('buffer_factor')}, "
                            f"sigma_clip={meta.get('sigma_clip')})"
                        )
                    else:
                        base_row["warnings"].append(
                            f"Applied {variant.detrender} (window={meta.get('window_cadences', 'N/A')} cadences)"
                        )
            t_var, f_var, fe_var, n_removed_non_finite = _clean_finite_inputs(t_var, f_var, fe_var)
            if n_removed_non_finite > 0:
                base_row["warnings"].append(
                    f"Removed {n_removed_non_finite} non-finite points before scoring"
                )
            if len(t_var) < 50:
                base_row["status"] = "failed"
                base_row["failure_reason"] = (
                    f"Insufficient points after transforms ({len(t_var)} < 50)"
                )
            else:
                score, depth_hat_ppm, depth_err_ppm = _score_variant(
                    time=t_var,
                    flux=f_var,
                    flux_err=fe_var,
                    period_days=float(period_days),
                    t0_btjd=float(t0_btjd),
                    duration_hours=float(duration_hours),
                    config=config,
                )
                if not _score_outputs_are_finite(
                    score=float(score),
                    depth_hat_ppm=float(depth_hat_ppm),
                    depth_err_ppm=float(depth_err_ppm),
                ):
                    base_row["status"] = "failed"
                    base_row["failure_reason"] = (
                        "non-finite score outputs "
                        f"(score={score}, depth_hat_ppm={depth_hat_ppm}, depth_err_ppm={depth_err_ppm})"
                    )
                else:
                    base_row["score"] = float(score)
                    base_row["depth_hat_ppm"] = float(depth_hat_ppm)
                    base_row["depth_err_ppm"] = float(depth_err_ppm)
                    base_row["n_points_used"] = int(len(t_var))
        except Exception as e:
            base_row["status"] = "failed"
            base_row["failure_reason"] = f"{type(e).__name__}: {e}"

        base_row["runtime_seconds"] = float(time_module.time() - start)
        sweep_table.append(SweepRow(**base_row))

    ok_rows = [r for r in sweep_table if r.status == "ok" and r.score is not None]
    n_ok = int(len(ok_rows))
    n_failed = int(len(sweep_table) - n_ok)

    score_spread: float | None = None
    depth_spread: float | None = None
    metric_variance: float | None = None
    stable = False
    best_variant_id: str | None = None
    worst_variant_id: str | None = None

    if n_ok >= 3:
        scores = np.array([abs(float(r.score)) for r in ok_rows], dtype=np.float64)
        depths = np.array([abs(float(r.depth_hat_ppm or 0.0)) for r in ok_rows], dtype=np.float64)

        if len(scores) >= 3:
            median = float(np.median(scores))
            q1, q3 = float(np.percentile(scores, 25)), float(np.percentile(scores, 75))
            score_spread = float((q3 - q1) / max(median, 1e-12))
        if len(depths) >= 3:
            median = float(np.median(depths))
            q1, q3 = float(np.percentile(depths, 25)), float(np.percentile(depths, 75))
            depth_spread = float((q3 - q1) / max(median, 1e-12))

        if score_spread is not None and depth_spread is not None:
            metric_variance = float(max(score_spread, depth_spread))
        elif score_spread is not None:
            metric_variance = float(score_spread)
        elif depth_spread is not None:
            metric_variance = float(depth_spread)

        if metric_variance is not None:
            stable = bool(metric_variance <= float(stability_threshold))

        score_by_id = {r.variant_id: abs(float(r.score)) for r in ok_rows if r.score is not None}
        if score_by_id:
            best_variant_id = max(score_by_id, key=lambda k: score_by_id[k])
            worst_variant_id = min(score_by_id, key=lambda k: score_by_id[k])
    else:
        notes.append(f"Insufficient ok variants ({n_ok}) to compute stability metrics (need >= 3)")

    return SensitivitySweepResult(
        stable=bool(stable),
        metric_variance=metric_variance,
        score_spread_iqr_over_median=score_spread,
        depth_spread_iqr_over_median=depth_spread,
        n_variants_total=int(len(sweep_table)),
        n_variants_ok=n_ok,
        n_variants_failed=n_failed,
        best_variant_id=best_variant_id,
        worst_variant_id=worst_variant_id,
        stability_threshold=float(stability_threshold),
        notes=notes,
        sweep_table=sweep_table,
    )
