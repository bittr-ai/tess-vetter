"""TLS calibration CLI (subprocess-safe).

Invoked as `python -m tess_vetter.cli.mlx_tls_calibration_cli -- ...`.

This script intentionally avoids host-specific storage layers; it returns a JSON
payload to stdout and lets the host (e.g. an MCP tool) decide whether/how to
persist artifacts.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import warnings
from io import StringIO

import numpy as np

from tess_vetter.api.io import PersistentCache
from tess_vetter.compute.detrend import bin_median_trend


def _import_tls() -> tuple[object, object]:
    try:
        from transitleastsquares import transitleastsquares  # type: ignore[import-not-found]
        from transitleastsquares.grid import period_grid  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "This CLI requires the 'tls' extra. Install with: pip install 'tess-vetter[tls]'"
        ) from e
    return transitleastsquares, period_grid


def _transit_mask(
    time_np: np.ndarray, period: float, t0: float, duration_hours: float
) -> np.ndarray:
    duration_days = duration_hours / 24.0
    phase = ((time_np - t0) / period) % 1.0
    phase = np.minimum(phase, 1.0 - phase)
    dt = phase * period
    return dt <= (duration_days / 2.0)


def _phase_diff_days(t_ref: float, t_test: float, period: float) -> float:
    dt = (t_test - t_ref) % period
    if dt > period / 2.0:
        dt -= period
    return float(dt)


def _choose_oversampling_factor(
    *, time_span_days: float, period_min: float, period_max: float, min_grid: int = 100
) -> tuple[int, np.ndarray]:
    _, period_grid = _import_tls()
    o = 3
    while o <= 320:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            periods = period_grid(
                R_star=1.0,
                M_star=1.0,
                time_span=time_span_days,
                period_min=period_min,
                period_max=period_max,
                oversampling_factor=o,
                n_transits_min=2,
            )
        if (
            len(periods) >= min_grid
            and float(np.min(periods)) >= period_min
            and float(np.max(periods)) <= period_max
        ):
            return o, periods
        o *= 2
    return 320, periods


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _smooth_box_template_np(
    time_np: np.ndarray,
    *,
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    ingress_egress_fraction: float = 0.2,
    sharpness: float = 30.0,
) -> np.ndarray:
    period = float(period_days)
    t0 = float(t0_btjd)
    duration_days = float(duration_hours / 24.0)

    phase = (time_np - t0) / period
    phase = phase - np.floor(phase + 0.5)
    dt = np.abs(phase * period)

    half_duration = duration_days / 2.0
    ingress = max(duration_days * float(ingress_egress_fraction), 1e-6)
    k = float(sharpness) / ingress
    template = _sigmoid(k * (half_duration - dt))
    return np.clip(template, 0.0, 1.0).astype(np.float64)


def main(argv: list[str] | None = None) -> int:
    transitleastsquares, _ = _import_tls()

    parser = argparse.ArgumentParser(description="Run MLX vs TLS calibration (subprocess-safe).")
    parser.add_argument("data_ref")
    parser.add_argument("period", type=float)
    parser.add_argument("t0", type=float)
    parser.add_argument("duration_hours", type=float)
    parser.add_argument("depths_ppm_json")
    parser.add_argument("period_window_frac", type=float)
    parser.add_argument("tls_snr_threshold", type=float)
    parser.add_argument("remove_existing_signal", type=str)
    parser.add_argument("mlx_target_score", type=float)
    parser.add_argument("downsample_factor", type=int)
    parser.add_argument("max_points", type=int)
    parser.add_argument("baseline_detrend", type=str)
    parser.add_argument("detrend_bin_hours", type=float)
    args = parser.parse_args(argv)

    data_ref = args.data_ref
    period = float(args.period)
    t0 = float(args.t0)
    duration_hours = float(args.duration_hours)
    depths_ppm = json.loads(args.depths_ppm_json)
    window_frac = float(args.period_window_frac)
    tls_snr_threshold = float(args.tls_snr_threshold)
    remove_existing = str(args.remove_existing_signal).lower() == "true"
    mlx_target_score = float(args.mlx_target_score)
    downsample_factor = int(args.downsample_factor)
    max_points = int(args.max_points)
    baseline_detrend = str(args.baseline_detrend).lower() == "true"
    detrend_bin_hours = float(args.detrend_bin_hours)

    cache = PersistentCache.get_default()
    lc = cache.get(data_ref)
    if lc is None:
        print(json.dumps({"error": "Data not found in cache", "data_ref": data_ref}))
        return 0

    mask = lc.valid_mask
    time_np = lc.time[mask].astype(np.float64)
    flux_np = lc.flux[mask].astype(np.float64)
    ferr_np = lc.flux_err[mask].astype(np.float64)

    if time_np.size > max_points:
        stride = int(np.ceil(time_np.size / max_points))
        time_np = time_np[::stride]
        flux_np = flux_np[::stride]
        ferr_np = ferr_np[::stride]

    if downsample_factor > 1:
        time_np = time_np[::downsample_factor]
        flux_np = flux_np[::downsample_factor]
        ferr_np = ferr_np[::downsample_factor]

    detrend_warning: str | None = None
    baseline_detrend_applied = False
    if baseline_detrend:
        try:
            trend = bin_median_trend(
                time_np, flux_np, bin_hours=detrend_bin_hours, min_bin_points=1
            )
            trend = np.where(trend <= 0, np.nanmedian(trend[trend > 0]), trend)
            flux_np = flux_np / trend
            ferr_np = ferr_np / trend
            baseline_detrend_applied = True
        except Exception as e:
            detrend_warning = f"{type(e).__name__}: {e}"
            baseline_detrend_applied = False

    baseline_flux = flux_np.copy()
    if remove_existing:
        in_tr = _transit_mask(time_np, period, t0, duration_hours)
        baseline_level = np.nanmedian(baseline_flux)
        baseline_flux[in_tr] = baseline_level

    template_np = _smooth_box_template_np(
        time_np,
        period_days=period,
        t0_btjd=t0,
        duration_hours=duration_hours,
    )
    w = 1.0 / np.maximum(ferr_np * ferr_np, 1e-12)
    denom = float(np.nansum(w * template_np * template_np) + 1e-12)
    depth_sigma = float(np.sqrt(1.0 / denom))
    mlx_depth_threshold_ppm = depth_sigma * 1e6 * mlx_target_score

    results: list[dict[str, object]] = []
    pmin = period * (1.0 - window_frac)
    pmax = period * (1.0 + window_frac)
    time_span_days = float(time_np.max() - time_np.min())

    oversampling_factor, _ = _choose_oversampling_factor(
        time_span_days=time_span_days,
        period_min=pmin,
        period_max=pmax,
        min_grid=100,
    )

    duration_days = duration_hours / 24.0
    t0_tol_days = max(0.5 * duration_days, 0.01 * period)

    for dppm in depths_ppm:
        inj_depth_ppm = float(dppm)
        depth = inj_depth_ppm / 1e6
        flux_inj = baseline_flux - depth * template_np

        model = transitleastsquares(time_np, flux_inj, ferr_np, verbose=False)
        with contextlib.redirect_stdout(StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tls_res = model.power(
                period_min=pmin,
                period_max=pmax,
                oversampling_factor=oversampling_factor,
                n_transits_min=2,
                show_progress_bar=False,
                use_threads=1,
                verbose=False,
            )

        tls_snr = float(getattr(tls_res, "SDE", np.nan))
        tls_period = float(getattr(tls_res, "period", np.nan))
        tls_t0 = float(getattr(tls_res, "T0", np.nan))
        tls_fap = float(getattr(tls_res, "FAP", np.nan))

        # Ephemeris consistency: period close + phase close (modulo P)
        period_ok = np.isfinite(tls_period) and abs(tls_period - period) <= max(0.01 * period, 0.05)
        phase_diff = (
            _phase_diff_days(t_ref=t0, t_test=tls_t0, period=period)
            if np.isfinite(tls_t0)
            else np.nan
        )
        t0_ok = np.isfinite(phase_diff) and abs(phase_diff) <= t0_tol_days
        eph_ok = bool(period_ok and t0_ok)
        recovered = bool(eph_ok and np.isfinite(tls_snr) and tls_snr >= tls_snr_threshold)

        results.append(
            {
                "depth_ppm": float(inj_depth_ppm),
                "tls_snr": float(tls_snr),
                "tls_period": float(tls_period),
                "tls_fap": float(tls_fap),
                "ephemeris_consistent": bool(eph_ok),
                "recovered": bool(recovered),
            }
        )

    results = sorted(results, key=lambda r: float(r["depth_ppm"]))
    tls_threshold_ppm = None
    for r in results:
        if float(r["depth_ppm"]) > 0.0 and float(r.get("tls_snr") or 0.0) >= tls_snr_threshold:
            tls_threshold_ppm = float(r["depth_ppm"])
            break

    tls_threshold_ppm_ephemeris_consistent = None
    for r in results:
        if (
            float(r["depth_ppm"]) > 0.0
            and bool(r.get("ephemeris_consistent"))
            and float(r.get("tls_snr") or 0.0) >= tls_snr_threshold
        ):
            tls_threshold_ppm_ephemeris_consistent = float(r["depth_ppm"])
            break

    strict_rows = [
        r
        for r in results
        if float(r["depth_ppm"]) > 0.0
        and bool(r.get("ephemeris_consistent"))
        and np.isfinite(float(r.get("tls_snr", np.nan)))
    ]
    strict_rows = sorted(strict_rows, key=lambda x: float(x["depth_ppm"]))
    tls_threshold_ppm_est = None
    if len(strict_rows) >= 2:
        below = [r for r in strict_rows if float(r["tls_snr"]) < tls_snr_threshold]
        above = [r for r in strict_rows if float(r["tls_snr"]) >= tls_snr_threshold]
        if below and above:
            lo = below[-1]
            hi = above[0]
            dlo = float(lo["depth_ppm"])
            dhi = float(hi["depth_ppm"])
            slo = float(lo["tls_snr"])
            shi = float(hi["tls_snr"])
            if dhi > dlo and shi != slo:
                frac = (tls_snr_threshold - slo) / (shi - slo)
                est = dlo + frac * (dhi - dlo)
                if np.isfinite(est):
                    tls_threshold_ppm_est = float(max(0.0, est))

    out = {
        "tic_id": int(lc.tic_id),
        "sector": int(lc.sector),
        "n_valid_points": int(mask.sum()),
        "baseline_detrend_requested": bool(baseline_detrend),
        "baseline_detrend_applied": bool(baseline_detrend_applied),
        "baseline_detrend_warning": detrend_warning,
        "detrend_bin_hours": float(detrend_bin_hours),
        "mlx_depth_threshold_ppm": float(mlx_depth_threshold_ppm),
        "tls_threshold_ppm": tls_threshold_ppm,
        "tls_threshold_ppm_ephemeris_consistent": tls_threshold_ppm_ephemeris_consistent,
        "tls_threshold_ppm_est": tls_threshold_ppm_est,
        "tls_snr_threshold": float(tls_snr_threshold),
        "period_window_frac": float(window_frac),
        "remove_existing_signal": bool(remove_existing),
        "tls_oversampling_factor": int(oversampling_factor),
        "results": results,
    }
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    import sys

    if "--" in sys.argv:
        idx = sys.argv.index("--")
        args = sys.argv[idx + 1 :]
    else:
        args = sys.argv[1:]
    raise SystemExit(main(args))
