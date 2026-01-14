"""MLX BLS-like search CLI (range mode, subprocess-safe).

Invoked as `python -m bittr_tess_vetter.cli.mlx_bls_search_range_cli -- ...`.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from bittr_tess_vetter.api.io import PersistentCache
from bittr_tess_vetter.cli.mlx_bls_search_cli import bls_like_search_mlx


@dataclass(frozen=True)
class RangeSearchRequest:
    data_ref: str
    period_min: float
    period_max: float
    periods: int
    durations_hours: list[float]
    downsample: int
    max_points: int
    n_t0_phases: int
    chunk_periods: int
    use_flux_err: bool
    top_k: int


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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="MLX GPU BLS-like range search (subprocess-safe).")
    ap.add_argument("data_ref")
    ap.add_argument("period_min", type=float)
    ap.add_argument("period_max", type=float)
    ap.add_argument("periods", type=int)
    ap.add_argument("durations_hours_json", type=str)
    ap.add_argument("downsample", type=int)
    ap.add_argument("max_points", type=int)
    ap.add_argument("n_t0_phases", type=int)
    ap.add_argument("chunk_periods", type=int)
    ap.add_argument("use_flux_err", type=str)
    ap.add_argument("top_k", type=int)
    args = ap.parse_args(argv)

    try:
        durations = [float(x) for x in json.loads(args.durations_hours_json)]
    except Exception as e:
        print(json.dumps({"error": "invalid_durations_hours_json", "message": str(e)}))
        return 0

    req = RangeSearchRequest(
        data_ref=args.data_ref,
        period_min=float(args.period_min),
        period_max=float(args.period_max),
        periods=int(args.periods),
        durations_hours=durations,
        downsample=int(args.downsample),
        max_points=int(args.max_points),
        n_t0_phases=int(args.n_t0_phases),
        chunk_periods=int(args.chunk_periods),
        use_flux_err=str(args.use_flux_err).lower() in {"1", "true", "yes"},
        top_k=int(args.top_k),
    )

    if not (req.period_max > req.period_min > 0):
        print(
            json.dumps(
                {
                    "error": "invalid_period_range",
                    "period_min": req.period_min,
                    "period_max": req.period_max,
                }
            )
        )
        return 0
    if req.periods < 2:
        print(json.dumps({"error": "invalid_periods", "periods": req.periods}))
        return 0

    try:
        start = time.perf_counter()
        time_arr, flux_arr, ferr_arr = _load_lc_arrays(req.data_ref)

        if req.downsample > 1:
            time_arr = time_arr[:: req.downsample]
            flux_arr = flux_arr[:: req.downsample]
            ferr_arr = ferr_arr[:: req.downsample]

        if req.max_points > 0 and time_arr.size > req.max_points:
            keep = int(req.max_points)
            idx = np.linspace(0, time_arr.size - 1, keep).astype(int)
            time_arr = time_arr[idx]
            flux_arr = flux_arr[idx]
            ferr_arr = ferr_arr[idx]

        period_grid = np.linspace(req.period_min, req.period_max, req.periods, dtype=np.float64)

        res, candidates = bls_like_search_mlx(
            time_btjd=time_arr,
            flux=flux_arr,
            flux_err=ferr_arr,
            period_grid=period_grid,
            duration_hours_grid=req.durations_hours,
            n_t0_phases=req.n_t0_phases,
            chunk_periods=req.chunk_periods,
            use_flux_err=req.use_flux_err,
            top_k=req.top_k,
        )
        end = time.perf_counter()

        out: dict[str, Any] = {
            "data_ref": req.data_ref,
            "n_points_used": int(time_arr.size),
            "period_range": [float(req.period_min), float(req.period_max)],
            "periods": int(req.periods),
            "mlx_bls_like": res.__dict__,
            "candidates": [c.__dict__ for c in candidates],
            "runtime_seconds_total": float(end - start),
        }
        print(json.dumps(out))
        return 0
    except Exception as e:
        print(
            json.dumps(
                {"error": "mlx_bls_range_failed", "message": str(e), "data_ref": req.data_ref}
            )
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
