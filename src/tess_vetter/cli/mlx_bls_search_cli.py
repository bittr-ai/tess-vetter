"""MLX BLS-like search CLI (subprocess-safe).

Invoked as `python -m tess_vetter.cli.mlx_bls_search_cli -- ...`.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from tess_vetter.api.io import PersistentCache


def _try_import_mlx() -> Any:
    try:
        import mlx.core as mx  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "MLX is not installed or failed to import. Install (Apple Silicon): `pip install mlx`."
        ) from e
    return mx


@dataclass(frozen=True)
class MlxBlsLikeResult:
    method: str
    best_period_days: float
    best_t0_btjd: float
    best_duration_hours: float
    score: float
    runtime_seconds: float
    notes: dict[str, Any]


@dataclass(frozen=True)
class MlxBlsLikeCandidate:
    period_days: float
    t0_btjd: float
    duration_hours: float
    score: float


def _load_lc_arrays(data_ref: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache = PersistentCache.get_default()
    lc = cache.get(data_ref)
    if lc is None:
        raise SystemExit(f"data_ref not in cache: {data_ref}")
    m = lc.valid_mask
    return (
        lc.time[m].astype(np.float64),
        lc.flux[m].astype(np.float64),
        lc.flux_err[m].astype(np.float64),
    )


def _make_t0_grid_for_period(
    mx: Any,
    *,
    period_days: Any,  # mx scalar
    time_mid_btjd: Any,  # mx scalar
    n_t0_phases: int,
) -> Any:
    phases = mx.arange(n_t0_phases) / mx.array(float(n_t0_phases))
    t0_mod = phases * period_days
    k = mx.round((time_mid_btjd - t0_mod) / period_days)
    return t0_mod + k * period_days


def _score_given_t0s(
    mx: Any,
    *,
    time: Any,  # (N,)
    flux: Any,  # (N,)
    flux_err: Any | None,  # (N,) or None
    period_days: Any,  # scalar
    t0s_btjd: Any,  # (M,)
    duration_hours: Any,  # scalar
    ingress_egress_fraction: float = 0.2,
    sharpness: float = 30.0,
    eps: float = 1e-12,
) -> Any:
    period = period_days
    duration_days = duration_hours / mx.array(24.0)
    half_duration = duration_days / mx.array(2.0)

    phase = (time[:, None] - t0s_btjd[None, :]) / period
    phase = phase - mx.floor(phase + mx.array(0.5))
    dt = mx.abs(phase * period)

    ingress = mx.maximum(duration_days * mx.array(ingress_egress_fraction), mx.array(1e-6))
    k = mx.array(float(sharpness)) / ingress
    template = mx.sigmoid(k * (half_duration - dt))
    template = mx.clip(template, 0.0, 1.0)

    y = mx.array(1.0) - flux
    if flux_err is None:
        w = mx.ones_like(y)
    else:
        sigma2 = mx.maximum(flux_err * flux_err, mx.array(eps))
        w = mx.array(1.0) / sigma2

    denom = mx.sum(w[:, None] * template * template, axis=0) + mx.array(eps)
    depth_hat = mx.sum(w[:, None] * template * y[:, None], axis=0) / denom
    depth_sigma = mx.sqrt(mx.array(1.0) / denom)
    return depth_hat / mx.maximum(depth_sigma, mx.array(eps))


def bls_like_search_mlx(
    *,
    time_btjd: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    period_grid: np.ndarray,
    duration_hours_grid: list[float],
    n_t0_phases: int = 128,
    local_refine_steps: int = 11,
    local_refine_width_phase: float = 0.02,
    chunk_periods: int = 32,
    use_flux_err: bool = True,
    top_k: int = 10,
) -> tuple[MlxBlsLikeResult, list[MlxBlsLikeCandidate]]:
    mx = _try_import_mlx()
    start = time.perf_counter()

    if time_btjd.size < 10:
        raise ValueError("Need at least 10 points")
    if len(duration_hours_grid) == 0:
        raise ValueError("Need at least one duration_hours value")
    if chunk_periods <= 0:
        raise ValueError("chunk_periods must be >= 1")

    time_mx = mx.array(time_btjd.astype(np.float32))
    flux_mx = mx.array(flux.astype(np.float32))
    ferr_mx = mx.array(flux_err.astype(np.float32)) if use_flux_err else None
    t_mid_mx = mx.array(float(np.nanmedian(time_btjd)), dtype=mx.float32)

    durations = [float(d) for d in duration_hours_grid]
    scan_phases = np.linspace(
        -local_refine_width_phase, local_refine_width_phase, local_refine_steps
    ).astype(np.float32)
    scan_phases_mx = mx.array(scan_phases)

    if top_k <= 0:
        raise ValueError("top_k must be >= 1")

    best_score = float("-inf")
    best_period = float("nan")
    best_t0 = float("nan")
    best_dur = float("nan")

    top: list[MlxBlsLikeCandidate] = []

    periods = period_grid.astype(np.float32)
    n_periods = int(periods.size)
    for i0 in range(0, n_periods, int(chunk_periods)):
        chunk = periods[i0 : i0 + int(chunk_periods)]
        periods_mx = mx.array(chunk)

        def _best_for_period(p: Any) -> Any:
            t0_grid = _make_t0_grid_for_period(
                mx, period_days=p, time_mid_btjd=t_mid_mx, n_t0_phases=int(n_t0_phases)
            )
            best_s = mx.array(-1e9)
            best_t0_local = t0_grid[mx.array(0)]
            best_dur_local = mx.array(durations[0])

            for dur_h in durations:
                dur_mx = mx.array(float(dur_h))
                scores_t0 = _score_given_t0s(
                    mx,
                    time=time_mx,
                    flux=flux_mx,
                    flux_err=ferr_mx,
                    period_days=p,
                    t0s_btjd=t0_grid,
                    duration_hours=dur_mx,
                )
                idx = mx.argmax(scores_t0)
                s = mx.max(scores_t0)
                t0_pick = t0_grid[idx]

                better = s > best_s
                best_s = mx.where(better, s, best_s)
                best_t0_local = mx.where(better, t0_pick, best_t0_local)
                best_dur_local = mx.where(better, dur_mx, best_dur_local)

            t0_scan = best_t0_local + scan_phases_mx * p
            scores_scan = _score_given_t0s(
                mx,
                time=time_mx,
                flux=flux_mx,
                flux_err=ferr_mx,
                period_days=p,
                t0s_btjd=t0_scan,
                duration_hours=best_dur_local,
            )
            idx2 = mx.argmax(scores_scan)
            s2 = mx.max(scores_scan)
            t0_2 = t0_scan[idx2]

            better2 = s2 > best_s
            best_s = mx.where(better2, s2, best_s)
            best_t0_local = mx.where(better2, t0_2, best_t0_local)

            return mx.stack([best_s, best_t0_local, best_dur_local])

        chunk_res = mx.vmap(_best_for_period)(periods_mx)
        mx.eval(chunk_res)
        chunk_scores = chunk_res[:, 0]
        j = mx.argmax(chunk_scores)
        s_best = chunk_scores[j]
        p_best = periods_mx[j]
        t0_best_mx = chunk_res[j, 1]
        dur_best_mx = chunk_res[j, 2]
        mx.eval(s_best, p_best, t0_best_mx, dur_best_mx)

        s_val = float(s_best.item())
        if s_val > best_score:
            best_score = s_val
            best_period = float(p_best.item())
            best_t0 = float(t0_best_mx.item())
            best_dur = float(dur_best_mx.item())

        chunk_res_np = np.array(chunk_res, dtype=np.float32)
        for row, p in zip(chunk_res_np, chunk, strict=False):
            s = float(row[0])
            if not np.isfinite(s):
                continue
            top.append(
                MlxBlsLikeCandidate(
                    period_days=float(p),
                    t0_btjd=float(row[1]),
                    duration_hours=float(row[2]),
                    score=float(s),
                )
            )
        top.sort(key=lambda c: c.score, reverse=True)
        if len(top) > int(top_k) * 4:
            top = top[: int(top_k) * 4]

    end = time.perf_counter()
    top.sort(key=lambda c: c.score, reverse=True)
    top = top[: min(int(top_k), len(top))]
    return (
        MlxBlsLikeResult(
            method="mlx_bls_like",
            best_period_days=float(best_period),
            best_t0_btjd=float(best_t0),
            best_duration_hours=float(best_dur),
            score=float(best_score),
            runtime_seconds=float(end - start),
            notes={
                "n_t0_phases": int(n_t0_phases),
                "local_refine_steps": int(local_refine_steps),
                "local_refine_width_phase": float(local_refine_width_phase),
                "chunk_periods": int(chunk_periods),
                "n_periods": int(period_grid.size),
                "n_durations": int(len(duration_hours_grid)),
                "use_flux_err": bool(use_flux_err),
                "dtype": "float32",
                "top_k": int(top_k),
            },
        ),
        top,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="MLX GPU BLS-like search (subprocess-safe).")
    ap.add_argument("data_ref")
    ap.add_argument("period_center", type=float)
    ap.add_argument("period_window_frac", type=float)
    ap.add_argument("periods", type=int)
    ap.add_argument("durations_hours_json", type=str)
    ap.add_argument("downsample", type=int)
    ap.add_argument("n_t0_phases", type=int)
    ap.add_argument("chunk_periods", type=int)
    ap.add_argument("use_flux_err", type=str)
    ap.add_argument("max_points", type=int)
    ap.add_argument("top_k", type=int)
    args = ap.parse_args(argv)

    try:
        durations = [float(x) for x in json.loads(args.durations_hours_json)]
    except Exception as e:
        raise SystemExit(f"Invalid durations_hours_json: {e}") from e

    time_arr, flux_arr, ferr_arr = _load_lc_arrays(args.data_ref)
    if args.downsample > 1:
        time_arr = time_arr[:: args.downsample]
        flux_arr = flux_arr[:: args.downsample]
        ferr_arr = ferr_arr[:: args.downsample]

    if args.max_points > 0 and time_arr.size > args.max_points:
        keep = int(args.max_points)
        idx = np.linspace(0, time_arr.size - 1, keep).astype(int)
        time_arr = time_arr[idx]
        flux_arr = flux_arr[idx]
        ferr_arr = ferr_arr[idx]

    pmin = float(args.period_center) * (1.0 - float(args.period_window_frac))
    pmax = float(args.period_center) * (1.0 + float(args.period_window_frac))
    period_grid = np.linspace(pmin, pmax, int(args.periods), dtype=np.float64)

    try:
        res, candidates = bls_like_search_mlx(
            time_btjd=time_arr,
            flux=flux_arr,
            flux_err=ferr_arr,
            period_grid=period_grid,
            duration_hours_grid=durations,
            n_t0_phases=int(args.n_t0_phases),
            chunk_periods=int(args.chunk_periods),
            use_flux_err=str(args.use_flux_err).lower() in {"1", "true", "yes"},
            top_k=int(args.top_k),
        )
        out: dict[str, Any] = {
            "data_ref": args.data_ref,
            "n_points": int(time_arr.size),
            "period_window": [float(pmin), float(pmax)],
            "mlx_bls_like": res.__dict__,
            "candidates": [c.__dict__ for c in candidates],
        }
        print(json.dumps(out))
        return 0
    except Exception as e:
        err = {"error": "mlx_bls_like_failed", "message": str(e), "data_ref": args.data_ref}
        print(json.dumps(err))
        return 0


if __name__ == "__main__":
    import sys

    if "--" in sys.argv:
        idx = sys.argv.index("--")
        args = sys.argv[idx + 1 :]
    else:
        args = sys.argv[1:]
    raise SystemExit(main(args))
