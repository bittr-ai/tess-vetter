"""MLX refinement CLI (subprocess-safe).

Invoked as `python -m bittr_tess_vetter.cli.mlx_refine_candidates_cli -- ...`.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from bittr_tess_vetter.api.io import PersistentCache


def _try_import_mlx() -> Any:
    try:
        import mlx.core as mx  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "MLX is not installed or failed to import. Install (Apple Silicon): `pip install mlx`."
        ) from e
    return mx


@dataclass(frozen=True)
class Candidate:
    period_days: float
    t0_btjd: float
    duration_hours: float


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


def _refine_one(
    mx: Any,
    *,
    time_mx: Any,
    flux_mx: Any,
    ferr_mx: Any | None,
    candidate: Candidate,
    steps: int,
    lr: float,
    duration_bounds_mode: str,
    duration_min_hours: float,
    duration_max_hours: float,
    duration_min_factor: float,
    duration_max_factor: float,
    t0_window_phase: float,
    ingress_egress_fraction: float,
    sharpness: float,
) -> dict[str, Any]:
    period = mx.array(float(candidate.period_days))
    t0_init = mx.array(float(candidate.t0_btjd))

    if duration_bounds_mode not in {"absolute", "relative"}:
        raise ValueError("duration_bounds_mode must be 'absolute' or 'relative'")
    if duration_bounds_mode == "relative":
        dur_min_h = float(candidate.duration_hours) * float(duration_min_factor)
        dur_max_h = float(candidate.duration_hours) * float(duration_max_factor)
    else:
        dur_min_h = float(duration_min_hours)
        dur_max_h = float(duration_max_hours)

    # Clamp and validate
    dur_min_h = max(1e-3, min(dur_min_h, 48.0))
    dur_max_h = max(1e-3, min(dur_max_h, 48.0))
    if dur_max_h <= dur_min_h:
        raise ValueError("duration_max bound must be > duration_min bound")

    dur_min = mx.array(float(dur_min_h))
    dur_max = mx.array(float(dur_max_h))

    # Parameterization:
    # - t0: bounded offset around initial guess (prevents drift to unrelated epochs)
    # - duration: bounded in [min,max]
    delta_max_days = mx.array(float(t0_window_phase)) * period

    def _unpack(params: Any) -> tuple[Any, Any]:
        raw_dt0 = params[0]
        raw_dur = params[1]
        t0 = t0_init + delta_max_days * mx.tanh(raw_dt0)
        dur_hours = dur_min + (dur_max - dur_min) * mx.sigmoid(raw_dur)
        return t0, dur_hours

    def _score_depth_sigma(t0: Any, dur_hours: Any) -> tuple[Any, Any, Any]:
        duration_days = dur_hours / mx.array(24.0)
        half_duration = duration_days / mx.array(2.0)

        phase = (time_mx - t0) / period
        phase = phase - mx.floor(phase + mx.array(0.5))
        dt = mx.abs(phase * period)

        ingress = mx.maximum(
            duration_days * mx.array(float(ingress_egress_fraction)), mx.array(1e-6)
        )
        k = mx.array(float(sharpness)) / ingress
        template = mx.sigmoid(k * (half_duration - dt))
        template = mx.clip(template, 0.0, 1.0)

        y = mx.array(1.0) - flux_mx
        if ferr_mx is None:
            w = mx.ones_like(y)
        else:
            sigma2 = mx.maximum(ferr_mx * ferr_mx, mx.array(1e-12))
            w = mx.array(1.0) / sigma2

        denom = mx.sum(w * template * template) + mx.array(1e-12)
        depth_hat = mx.sum(w * template * y) / denom
        depth_sigma = mx.sqrt(mx.array(1.0) / denom)
        score = depth_hat / mx.maximum(depth_sigma, mx.array(1e-12))
        return score, depth_hat, depth_sigma

    def loss_fn(params: Any) -> Any:
        t0, dur_h = _unpack(params)
        score, _, _ = _score_depth_sigma(t0, dur_h)
        return -score

    grad_fn = mx.grad(loss_fn)

    params = mx.array([0.0, 0.0], dtype=mx.float32)
    m = mx.zeros_like(params)
    v = mx.zeros_like(params)
    beta1 = mx.array(0.9)
    beta2 = mx.array(0.999)
    eps = mx.array(1e-8)

    lr_mx = mx.array(float(lr))
    for i in range(int(steps)):
        g = grad_fn(params)
        m = beta1 * m + (mx.array(1.0) - beta1) * g
        v = beta2 * v + (mx.array(1.0) - beta2) * (g * g)
        t = mx.array(float(i + 1))
        m_hat = m / (mx.array(1.0) - mx.power(beta1, t))
        v_hat = v / (mx.array(1.0) - mx.power(beta2, t))
        params = params - lr_mx * m_hat / (mx.sqrt(v_hat) + eps)

    t0_final, dur_final_h = _unpack(params)
    score, depth_hat, depth_sigma = _score_depth_sigma(t0_final, dur_final_h)
    mx.eval(t0_final, dur_final_h, score, depth_hat, depth_sigma)

    depth_ppm = float(depth_hat.item()) * 1_000_000.0
    depth_sigma_ppm = float(depth_sigma.item()) * 1_000_000.0
    t0_shift_cycles = float(
        (t0_final.item() - float(candidate.t0_btjd)) / float(candidate.period_days)
    )
    t0_shift_cycles_wrapped = ((t0_shift_cycles + 0.5) % 1.0) - 0.5
    dur_ratio = float(dur_final_h.item()) / max(float(candidate.duration_hours), 1e-9)

    return {
        "period_days": float(candidate.period_days),
        "t0_init_btjd": float(candidate.t0_btjd),
        "duration_init_hours": float(candidate.duration_hours),
        "t0_refined_btjd": float(t0_final.item()),
        "duration_refined_hours": float(dur_final_h.item()),
        "t0_shift_cycles": float(t0_shift_cycles),
        "t0_shift_cycles_wrapped": float(t0_shift_cycles_wrapped),
        "duration_ratio_refined_over_init": float(dur_ratio),
        "depth_hat_ppm": float(depth_ppm),
        "depth_sigma_ppm": float(depth_sigma_ppm),
        "score_z": float(score.item()),
        "steps": int(steps),
        "lr": float(lr),
        "t0_window_phase": float(t0_window_phase),
        "duration_bounds_mode": str(duration_bounds_mode),
        "duration_bounds_hours": [float(dur_min_h), float(dur_max_h)],
        "ingress_egress_fraction": float(ingress_egress_fraction),
        "sharpness": float(sharpness),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Refine transit candidates using MLX (subprocess-safe)."
    )
    ap.add_argument("data_ref")
    ap.add_argument("candidates_json")
    ap.add_argument("steps", type=int)
    ap.add_argument("lr", type=float)
    ap.add_argument("duration_bounds_mode", type=str)
    ap.add_argument("duration_min_hours", type=float)
    ap.add_argument("duration_max_hours", type=float)
    ap.add_argument("duration_min_factor", type=float)
    ap.add_argument("duration_max_factor", type=float)
    ap.add_argument("t0_window_phase", type=float)
    ap.add_argument("ingress_egress_fraction", type=float)
    ap.add_argument("sharpness", type=float)
    ap.add_argument("downsample", type=int)
    ap.add_argument("max_points", type=int)
    ap.add_argument("use_flux_err", type=str)
    args = ap.parse_args(argv)

    try:
        raw = json.loads(args.candidates_json)
        candidates = [
            Candidate(
                period_days=float(c["period_days"]),
                t0_btjd=float(c["t0_btjd"]),
                duration_hours=float(c["duration_hours"]),
            )
            for c in raw
        ]
    except Exception as e:
        print(json.dumps({"error": "invalid_candidates_json", "message": str(e)}))
        return 0

    try:
        mx = _try_import_mlx()
        start = time.perf_counter()
        time_arr, flux_arr, ferr_arr = _load_lc_arrays(args.data_ref)

        if int(args.downsample) > 1:
            time_arr = time_arr[:: int(args.downsample)]
            flux_arr = flux_arr[:: int(args.downsample)]
            ferr_arr = ferr_arr[:: int(args.downsample)]

        if int(args.max_points) > 0 and time_arr.size > int(args.max_points):
            keep = int(args.max_points)
            idx = np.linspace(0, time_arr.size - 1, keep).astype(int)
            time_arr = time_arr[idx]
            flux_arr = flux_arr[idx]
            ferr_arr = ferr_arr[idx]

        time_mx = mx.array(time_arr.astype(np.float32))
        flux_mx = mx.array(flux_arr.astype(np.float32))
        use_err = str(args.use_flux_err).lower() in {"1", "true", "yes"}
        ferr_mx = mx.array(ferr_arr.astype(np.float32)) if use_err else None

        refined: list[dict[str, Any]] = []
        for c in candidates:
            refined.append(
                _refine_one(
                    mx,
                    time_mx=time_mx,
                    flux_mx=flux_mx,
                    ferr_mx=ferr_mx,
                    candidate=c,
                    steps=int(args.steps),
                    lr=float(args.lr),
                    duration_bounds_mode=str(args.duration_bounds_mode),
                    duration_min_hours=float(args.duration_min_hours),
                    duration_max_hours=float(args.duration_max_hours),
                    duration_min_factor=float(args.duration_min_factor),
                    duration_max_factor=float(args.duration_max_factor),
                    t0_window_phase=float(args.t0_window_phase),
                    ingress_egress_fraction=float(args.ingress_egress_fraction),
                    sharpness=float(args.sharpness),
                )
            )

        end = time.perf_counter()
        out = {
            "backend": "mlx",
            "data_ref": str(args.data_ref),
            "n_points_used": int(time_arr.size),
            "runtime_seconds": float(end - start),
            "refined": refined,
        }
        print(json.dumps(out))
        return 0
    except Exception as e:
        print(
            json.dumps({"error": "mlx_refine_failed", "message": str(e), "data_ref": args.data_ref})
        )
        return 0


if __name__ == "__main__":
    import sys

    if "--" in sys.argv:
        idx = sys.argv.index("--")
        args = sys.argv[idx + 1 :]
    else:
        args = sys.argv[1:]
    raise SystemExit(main(args))
