"""Light-curve-only systematics proxy features.

This module provides a small, deterministic set of "false-alarm / systematics"
features meant to flag transit-like artifacts even when pixel evidence is clean.

Design goals:
- LC-only (no attitude files required)
- Cheap to compute (suitable for bulk enrichment)
- Debuggable (returns a few interpretable subfeatures)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bittr_tess_vetter.validation.base import (
    get_in_transit_mask,
    get_out_of_transit_mask,
    measure_transit_depth,
)


def _robust_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 10:
        return float("nan")
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return float(1.4826 * mad)


@dataclass(frozen=True)
class SystematicsProxyResult:
    score: float
    few_point_top5_fraction: float | None
    max_step_sigma: float | None
    phase_shift_ratio_max: float | None
    inversion_same_sign: bool | None
    lag1_autocorr_oot: float | None
    n_in_transit: int
    n_out_of_transit: int

    def to_dict(self) -> dict[str, object]:
        return {
            "score": float(self.score),
            "few_point_top5_fraction": self.few_point_top5_fraction,
            "max_step_sigma": self.max_step_sigma,
            "phase_shift_ratio_max": self.phase_shift_ratio_max,
            "inversion_same_sign": self.inversion_same_sign,
            "lag1_autocorr_oot": self.lag1_autocorr_oot,
            "n_in_transit": int(self.n_in_transit),
            "n_out_of_transit": int(self.n_out_of_transit),
        }


def compute_systematics_proxy(
    *,
    time: np.ndarray,
    flux: np.ndarray,
    valid_mask: np.ndarray | None = None,
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
) -> SystematicsProxyResult | None:
    """Compute a minimal LC-only systematics proxy.

    Returns None when there isn't enough data to compute stable features.
    """
    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    if time.size != flux.size:
        return None

    if valid_mask is None:
        mask = np.isfinite(time) & np.isfinite(flux)
    else:
        vm = np.asarray(valid_mask, dtype=bool)
        if vm.shape != time.shape:
            return None
        mask = vm & np.isfinite(time) & np.isfinite(flux)

    time = time[mask]
    flux = flux[mask]

    if time.size < 200:
        return None

    in_mask = get_in_transit_mask(time, period_days, t0_btjd, duration_hours)
    out_mask = get_out_of_transit_mask(
        time, period_days, t0_btjd, duration_hours, buffer_factor=3.0
    )
    n_in = int(np.sum(in_mask))
    n_oot = int(np.sum(out_mask))

    # Default baseline/sigma fallback to global stats if OOT windows are too small.
    oot_flux = flux[out_mask] if n_oot >= 10 else flux
    baseline = (
        float(np.median(oot_flux[np.isfinite(oot_flux)])) if np.any(np.isfinite(oot_flux)) else 0.0
    )

    depth = None
    depth_abs = None
    if n_in >= 5 and n_oot >= 20:
        try:
            d, d_err = measure_transit_depth(flux, in_mask, out_mask)
            if np.isfinite(d) and np.isfinite(d_err) and d_err > 0:
                depth = float(d)
                depth_abs = float(abs(d))
        except Exception:
            depth = None
            depth_abs = None

    # Few-point dominance: fraction of the total depth contribution coming from the top 5 points.
    few_point_top5_fraction: float | None = None
    if depth_abs is not None and depth_abs > 0 and n_in >= 5:
        in_flux = flux[in_mask]
        deltas = baseline - in_flux
        deltas = np.asarray(deltas, dtype=np.float64)
        deltas = deltas[np.isfinite(deltas)]
        deltas_pos = deltas[deltas > 0]
    else:
        deltas_pos = np.array([], dtype=np.float64)

    if deltas_pos.size >= 10:
        # Use an adaptive top-k (20% of in-transit points, min 5) so real, well-sampled transits
        # do not look "few-point dominated" just because they are deep.
        k = int(max(5, np.ceil(0.2 * float(deltas_pos.size))))
        tot = float(np.sum(deltas_pos))
        if tot > 0:
            topk = float(np.sum(np.sort(deltas_pos)[-k:]))
            few_point_top5_fraction = float(np.clip(topk / tot, 0.0, 1.0))
        else:
            few_point_top5_fraction = 0.0

    # Step/discontinuity proxy: max consecutive step in units of oot robust sigma.
    _robust_sigma(oot_flux)
    max_step_sigma: float | None = None
    if int(np.sum(out_mask)) >= 10:
        # Step/discontinuity proxy should not be dominated by real transit edges.
        # Compute max step on *out-of-transit* samples, and ignore large cadence gaps.
        t_oot = time[out_mask]
        f_oot = flux[out_mask]
        if t_oot.size >= 3:
            order = np.argsort(t_oot)
            t_oot = t_oot[order]
            f_oot = f_oot[order]
            dt = np.diff(t_oot)
            df_signed = np.diff(f_oot)

            # Ignore diffs spanning large gaps (e.g., downlink gaps) that can inflate step metrics.
            med_dt = float(np.median(dt)) if dt.size else float("nan")
            if np.isfinite(med_dt) and med_dt > 0:
                ok = dt <= (5.0 * med_dt)
                df_signed = df_signed[ok]
            df_signed = df_signed[np.isfinite(df_signed)]

            # Use a *high-frequency* noise scale so large steps remain significant even if
            # the OOT distribution is bimodal (step inflates oot MAD).
            sigma_step = _robust_sigma(df_signed)
            if np.isfinite(sigma_step) and sigma_step > 0:
                max_step_sigma = float(np.max(np.abs(df_signed)) / float(sigma_step))

    # Phase-shift null (cheap): evaluate depth at a few off-phase offsets (in units of period).
    # A real transit should not persist at unrelated phases.
    phase_shift_ratio_max: float | None = None
    if depth_abs is not None and depth_abs > 0:
        try:
            shifts = [0.1 * float(period_days), 0.2 * float(period_days), 0.33 * float(period_days)]
            ratios: list[float] = []
            for dt in shifts:
                in_m = get_in_transit_mask(time, period_days, t0_btjd + dt, duration_hours)
                out_m = get_out_of_transit_mask(
                    time, period_days, t0_btjd + dt, duration_hours, buffer_factor=3.0
                )
                if int(np.sum(in_m)) < 5 or int(np.sum(out_m)) < 20:
                    continue
                d_shift, d_shift_err = measure_transit_depth(flux, in_m, out_m)
                if not (np.isfinite(d_shift) and np.isfinite(d_shift_err) and d_shift_err > 0):
                    continue
                ratios.append(float(abs(d_shift)) / depth_abs)
            if ratios:
                phase_shift_ratio_max = float(min(10.0, max(ratios)))
        except Exception:
            phase_shift_ratio_max = None

    # Inversion sign check: inverting the flux turns a dip into a bump, so the inferred depth should flip sign.
    inversion_same_sign: bool | None = None
    if depth is not None and n_in >= 5 and n_oot >= 20:
        try:
            flux_inv = 2.0 * baseline - flux
            d_inv, d_inv_err = measure_transit_depth(flux_inv, in_mask, out_mask)
            if np.isfinite(d_inv) and np.isfinite(d_inv_err) and d_inv_err > 0:
                inversion_same_sign = bool(np.sign(d_inv) == np.sign(depth))
        except Exception:
            inversion_same_sign = None

    # Lag-1 autocorrelation on OOT residuals (red-noise proxy).
    lag1_autocorr_oot: float | None = None
    try:
        r = oot_flux.astype(np.float64) - baseline
        r = r[np.isfinite(r)]
        if r.size >= 30:
            x = r[:-1]
            y = r[1:]
            sx = float(np.std(x))
            sy = float(np.std(y))
            if sx > 0 and sy > 0:
                lag1_autocorr_oot = float(np.corrcoef(x, y)[0, 1])
    except Exception:
        lag1_autocorr_oot = None

    # Combine into a 0..1 risk score (heuristic, intentionally simple).
    score = 0.0
    if few_point_top5_fraction is not None:
        if few_point_top5_fraction >= 0.7:
            score += 0.5
        elif few_point_top5_fraction >= 0.6:
            score += 0.3
        elif few_point_top5_fraction >= 0.5:
            score += 0.15

    if max_step_sigma is not None:
        if max_step_sigma >= 12.0:
            score += 0.5
        elif max_step_sigma >= 8.0:
            score += 0.3

    if phase_shift_ratio_max is not None:
        if phase_shift_ratio_max >= 0.7:
            score += 0.6
        elif phase_shift_ratio_max >= 0.5:
            score += 0.35
        elif phase_shift_ratio_max >= 0.3:
            score += 0.15

    if inversion_same_sign is True:
        score += 0.3

    if lag1_autocorr_oot is not None and lag1_autocorr_oot >= 0.5:
        score += 0.1

    score = float(np.clip(score, 0.0, 1.0))
    return SystematicsProxyResult(
        score=score,
        few_point_top5_fraction=few_point_top5_fraction,
        max_step_sigma=max_step_sigma,
        phase_shift_ratio_max=phase_shift_ratio_max,
        inversion_same_sign=inversion_same_sign,
        lag1_autocorr_oot=lag1_autocorr_oot,
        n_in_transit=n_in,
        n_out_of_transit=n_oot,
    )
