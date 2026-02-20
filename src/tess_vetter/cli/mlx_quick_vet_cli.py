"""MLX quick-vet CLI (subprocess-safe).

Invoked as `python -m tess_vetter.cli.mlx_quick_vet_cli -- ...`.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Literal

import numpy as np

from tess_vetter.api.io import PersistentCache


def _require_mlx() -> Any:
    try:
        import mlx.core as mx  # type: ignore[import-not-found]
    except Exception as e:
        raise ImportError(f"Failed to import mlx.core: {e}") from e
    return mx


def _template_batched(
    *,
    mx: Any,
    time: Any,  # (N,)
    period_days: float,
    t0s_btjd: Any,  # (K,)
    duration_hours: float,
    ingress_egress_fraction: float,
    sharpness: float,
) -> Any:  # (K,N)
    period = mx.array(float(period_days))
    duration_days = mx.array(float(duration_hours) / 24.0)

    half_duration = duration_days / 2.0
    ingress = mx.maximum(duration_days * mx.array(float(ingress_egress_fraction)), mx.array(1e-6))
    k = mx.array(float(sharpness)) / ingress

    phase = (time[None, :] - t0s_btjd[:, None]) / period
    phase = phase - mx.floor(phase + mx.array(0.5))
    dt = mx.abs(phase * period)
    tmpl = mx.sigmoid(k * (half_duration - dt))
    return mx.clip(tmpl, 0.0, 1.0)


def _batched_scores(
    *,
    mx: Any,
    time: Any,
    flux: Any,
    flux_err: Any,
    period_days: float,
    t0s_btjd: Any,
    duration_hours: float,
    ingress_egress_fraction: float,
    sharpness: float,
) -> tuple[Any, Any, Any]:
    y = mx.array(1.0) - flux
    w = mx.array(1.0) / mx.maximum(flux_err * flux_err, mx.array(1e-12))
    tmpl = _template_batched(
        mx=mx,
        time=time,
        period_days=period_days,
        t0s_btjd=t0s_btjd,
        duration_hours=duration_hours,
        ingress_egress_fraction=ingress_egress_fraction,
        sharpness=sharpness,
    )
    denom = mx.sum(w[None, :] * tmpl * tmpl, axis=1) + mx.array(1e-12)
    depth_hat = mx.sum(w[None, :] * tmpl * y[None, :], axis=1) / denom
    depth_sigma = mx.sqrt(mx.array(1.0) / denom)
    score = depth_hat / mx.maximum(depth_sigma, mx.array(1e-12))
    return score, depth_hat, depth_sigma


def _t0_shifts(
    *,
    t0: float,
    period: float,
    n: int,
    strategy: Literal["grid", "random"],
    seed: int,
) -> np.ndarray:
    if strategy == "random":
        rng = np.random.default_rng(int(seed))
        phases = rng.uniform(0.0, 1.0, size=int(n))
        phases = np.where(phases < 1e-3, phases + 1e-3, phases)
    else:
        phases = (np.arange(1, int(n) + 1, dtype=np.float64) / (float(n) + 1.0)).astype(np.float64)
    return (float(t0) + phases * float(period)).astype(np.float64)


def _estimate_curvature(t0s: np.ndarray, scores: np.ndarray, best_index: int) -> float:
    if best_index <= 0 or best_index >= len(t0s) - 1:
        return float("nan")
    x = t0s[best_index - 1 : best_index + 2]
    y = scores[best_index - 1 : best_index + 2]
    try:
        a, _b, _c = np.polyfit(x, y, 2)
    except Exception:
        return float("nan")
    return float(2.0 * a)


def _estimate_fwhm_minutes(t0s: np.ndarray, scores: np.ndarray, best_index: int) -> float:
    if len(t0s) < 5:
        return float("nan")
    peak = float(scores[best_index])
    baseline = float(np.median(scores))
    thresh = (peak + baseline) / 2.0

    left = best_index
    while left > 0 and float(scores[left]) >= thresh:
        left -= 1
    right = best_index
    while right < len(scores) - 1 and float(scores[right]) >= thresh:
        right += 1
    if left == best_index or right == best_index:
        return float("nan")
    width_days = float(t0s[right] - t0s[left])
    return float(width_days * 24.0 * 60.0)


def _load_arrays(
    *, data_ref: str, max_points: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    cache = PersistentCache.get_default()
    lc = cache.get(data_ref)
    if lc is None:
        raise ValueError(f"Data not found in cache: {data_ref}")

    mask = lc.valid_mask
    time = lc.time[mask].astype(np.float64)
    flux = lc.flux[mask].astype(np.float64)
    ferr = lc.flux_err[mask].astype(np.float64)

    if len(time) > int(max_points):
        idx = np.linspace(0, len(time) - 1, int(max_points), dtype=int)
        time = time[idx]
        flux = flux[idx]
        ferr = ferr[idx]

    return time, flux, ferr, int(len(time)), int(lc.tic_id), int(lc.sector)


def cmd_phase_shift_null(args: argparse.Namespace) -> dict[str, Any]:
    mx = _require_mlx()
    time_np, flux_np, ferr_np, n_used, tic_id, sector = _load_arrays(
        data_ref=args.data_ref, max_points=args.max_points
    )
    if n_used < 200:
        raise ValueError(f"Insufficient data points ({n_used})")

    time = mx.array(time_np.astype(float))
    flux = mx.array(flux_np.astype(float))
    ferr = mx.array(ferr_np.astype(float))

    t0_obs = mx.array(np.array([float(args.t0)], dtype=np.float64))
    obs_score, obs_depth_hat, obs_depth_sigma = _batched_scores(
        mx=mx,
        time=time,
        flux=flux,
        flux_err=ferr,
        period_days=float(args.period),
        t0s_btjd=t0_obs,
        duration_hours=float(args.duration_hours),
        ingress_egress_fraction=float(args.ingress_egress_fraction),
        sharpness=float(args.sharpness),
    )
    mx.eval(obs_score, obs_depth_hat, obs_depth_sigma)
    observed_score = float(obs_score[0])

    t0s_np = _t0_shifts(
        t0=float(args.t0),
        period=float(args.period),
        n=int(args.n_phase_shifts),
        strategy=str(args.phase_shift_strategy),
        seed=int(args.random_seed),
    )
    scores, _, _ = _batched_scores(
        mx=mx,
        time=time,
        flux=flux,
        flux_err=ferr,
        period_days=float(args.period),
        t0s_btjd=mx.array(t0s_np),
        duration_hours=float(args.duration_hours),
        ingress_egress_fraction=float(args.ingress_egress_fraction),
        sharpness=float(args.sharpness),
    )
    mx.eval(scores)

    null_mean = float(mx.mean(scores))
    centered = scores - mx.array(null_mean)
    null_std = float(mx.sqrt(mx.mean(centered * centered)))
    z_score = float((observed_score - null_mean) / null_std) if null_std > 0 else float("nan")

    n_ge = float(mx.sum(scores >= mx.array(observed_score)))
    p_value = (n_ge + 1.0) / (float(args.n_phase_shifts) + 1.0)

    return {
        "backend": "mlx",
        "observed_score": observed_score,
        "observed_depth_hat": float(obs_depth_hat[0]),
        "observed_depth_sigma": float(obs_depth_sigma[0]),
        "n_trials": int(args.n_phase_shifts),
        "strategy": str(args.phase_shift_strategy),
        "null_mean": null_mean,
        "null_std": null_std,
        "z_score": z_score,
        "p_value_one_sided": float(p_value),
        "n_used_points": n_used,
        "tic_id": tic_id,
        "sector": sector,
    }


def cmd_local_sensitivity(args: argparse.Namespace) -> dict[str, Any]:
    mx = _require_mlx()
    time_np, flux_np, ferr_np, n_used, _tic_id, _sector = _load_arrays(
        data_ref=args.data_ref, max_points=args.max_points
    )
    if n_used < 200:
        raise ValueError(f"Insufficient data points ({n_used})")

    half_span_days = float(args.half_span_minutes) / (24.0 * 60.0)
    t0s_np = (
        float(args.t0) + np.linspace(-half_span_days, half_span_days, int(args.n_grid))
    ).astype(np.float64)

    time = mx.array(time_np.astype(float))
    flux = mx.array(flux_np.astype(float))
    ferr = mx.array(ferr_np.astype(float))
    scores, _, _ = _batched_scores(
        mx=mx,
        time=time,
        flux=flux,
        flux_err=ferr,
        period_days=float(args.period),
        t0s_btjd=mx.array(t0s_np),
        duration_hours=float(args.duration_hours),
        ingress_egress_fraction=float(args.ingress_egress_fraction),
        sharpness=float(args.sharpness),
    )
    mx.eval(scores)
    scores_np = np.asarray(scores).astype(np.float64)

    best_index = int(np.argmax(scores_np))
    t0_best = float(t0s_np[best_index])
    score_best = float(scores_np[best_index])
    score_at_input = float(scores_np[int(np.argmin(np.abs(t0s_np - float(args.t0))))])

    curvature = _estimate_curvature(t0s_np, scores_np, best_index)
    fwhm_minutes = _estimate_fwhm_minutes(t0s_np, scores_np, best_index)

    return {
        "backend": "mlx",
        "t0_best_btjd": t0_best,
        "score_at_input": score_at_input,
        "score_best": score_best,
        "delta_score": float(score_best - score_at_input),
        "curvature": float(curvature),
        "fwhm_minutes": float(fwhm_minutes),
        "n_grid": int(args.n_grid),
        "half_span_minutes": float(args.half_span_minutes),
    }


def cmd_score(args: argparse.Namespace) -> dict[str, Any]:
    mx = _require_mlx()
    time_np, flux_np, ferr_np, n_used, tic_id, sector = _load_arrays(
        data_ref=args.data_ref, max_points=args.max_points
    )
    if n_used < 200:
        raise ValueError(f"Insufficient data points ({n_used})")

    time = mx.array(time_np.astype(float))
    flux = mx.array(flux_np.astype(float))
    ferr = mx.array(ferr_np.astype(float))

    t0_obs = mx.array(np.array([float(args.t0)], dtype=np.float64))
    score, depth_hat, depth_sigma = _batched_scores(
        mx=mx,
        time=time,
        flux=flux,
        flux_err=ferr,
        period_days=float(args.period),
        t0s_btjd=t0_obs,
        duration_hours=float(args.duration_hours),
        ingress_egress_fraction=float(args.ingress_egress_fraction),
        sharpness=float(args.sharpness),
    )
    mx.eval(score, depth_hat, depth_sigma)
    return {
        "backend": "mlx",
        "score": float(score[0]),
        "depth_hat": float(depth_hat[0]),
        "depth_sigma": float(depth_sigma[0]),
        "depth_hat_ppm": float(depth_hat[0]) * 1e6,
        "depth_sigma_ppm": float(depth_sigma[0]) * 1e6,
        "n_used_points": n_used,
        "tic_id": tic_id,
        "sector": sector,
    }


def cmd_score_refined(args: argparse.Namespace) -> dict[str, Any]:
    """Local t0 refinement: scan a small window and return the best score."""
    mx = _require_mlx()
    time_np, flux_np, ferr_np, n_used, tic_id, sector = _load_arrays(
        data_ref=args.data_ref, max_points=args.max_points
    )
    if n_used < 200:
        raise ValueError(f"Insufficient data points ({n_used})")

    time = mx.array(time_np.astype(float))
    flux = mx.array(flux_np.astype(float))
    ferr = mx.array(ferr_np.astype(float))

    half_span_minutes = args.half_span_minutes
    if half_span_minutes is None:
        half_span_minutes = float(min(120.0, max(10.0, 0.5 * float(args.duration_hours) * 60.0)))

    half_span_days = float(half_span_minutes) / (24.0 * 60.0)
    t0s_np = (
        float(args.t0) + np.linspace(-half_span_days, half_span_days, int(args.n_grid))
    ).astype(np.float64)

    scores, _, _ = _batched_scores(
        mx=mx,
        time=time,
        flux=flux,
        flux_err=ferr,
        period_days=float(args.period),
        t0s_btjd=mx.array(t0s_np),
        duration_hours=float(args.duration_hours),
        ingress_egress_fraction=float(args.ingress_egress_fraction),
        sharpness=float(args.sharpness),
    )
    mx.eval(scores)
    scores_np = np.asarray(scores).astype(np.float64)

    best_index = int(np.argmax(scores_np))
    t0_best = float(t0s_np[best_index])
    score_at_input = float(scores_np[int(np.argmin(np.abs(t0s_np - float(args.t0))))])

    # Compute depth_hat/depth_sigma at the best t0.
    t0_best_arr = mx.array(np.array([t0_best], dtype=np.float64))
    score_best, depth_hat, depth_sigma = _batched_scores(
        mx=mx,
        time=time,
        flux=flux,
        flux_err=ferr,
        period_days=float(args.period),
        t0s_btjd=t0_best_arr,
        duration_hours=float(args.duration_hours),
        ingress_egress_fraction=float(args.ingress_egress_fraction),
        sharpness=float(args.sharpness),
    )
    mx.eval(score_best, depth_hat, depth_sigma)

    return {
        "backend": "mlx",
        "score": float(score_best[0]),
        "score_at_input": float(score_at_input),
        "delta_score": float(float(score_best[0]) - float(score_at_input)),
        "t0_best_btjd": float(t0_best),
        "n_grid": int(args.n_grid),
        "half_span_minutes": float(half_span_minutes),
        "depth_hat": float(depth_hat[0]),
        "depth_sigma": float(depth_sigma[0]),
        "depth_hat_ppm": float(depth_hat[0]) * 1e6,
        "depth_sigma_ppm": float(depth_sigma[0]) * 1e6,
        "n_used_points": n_used,
        "tic_id": tic_id,
        "sector": sector,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m tess_vetter.cli.mlx_quick_vet_cli")
    sub = p.add_subparsers(dest="cmd", required=True)

    def _add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--data-ref", required=True)
        sp.add_argument("--period", type=float, required=True)
        sp.add_argument("--t0", type=float, required=True)
        sp.add_argument("--duration-hours", type=float, required=True)
        sp.add_argument("--max-points", type=int, default=50000)
        sp.add_argument("--ingress-egress-fraction", type=float, default=0.2)
        sp.add_argument("--sharpness", type=float, default=30.0)

    pnull = sub.add_parser("phase_shift_null")
    _add_common(pnull)
    pnull.add_argument("--n-phase-shifts", type=int, default=200)
    pnull.add_argument("--phase-shift-strategy", choices=["grid", "random"], default="grid")
    pnull.add_argument("--random-seed", type=int, default=0)
    pnull.set_defaults(func=cmd_phase_shift_null)

    plocal = sub.add_parser("local_sensitivity")
    _add_common(plocal)
    plocal.add_argument("--n-grid", type=int, default=81)
    plocal.add_argument("--half-span-minutes", type=float, required=True)
    plocal.set_defaults(func=cmd_local_sensitivity)

    pscore = sub.add_parser("score")
    _add_common(pscore)
    pscore.set_defaults(func=cmd_score)

    pscore_ref = sub.add_parser("score_refined")
    _add_common(pscore_ref)
    pscore_ref.add_argument("--n-grid", type=int, default=81)
    pscore_ref.add_argument(
        "--half-span-minutes",
        type=float,
        default=None,
        help="Half-width of the local t0 scan window in minutes (default: min(120, 0.5*duration))",
    )
    pscore_ref.set_defaults(func=cmd_score_refined)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        out = args.func(args)  # type: ignore[attr-defined]
        print(json.dumps(out))
        return 0
    except Exception as e:
        print(
            json.dumps(
                {
                    "error": type(e).__name__,
                    "message": str(e),
                }
            )
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
