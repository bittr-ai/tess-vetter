"""Pixel-localization facade for host applications.

This module consolidates the core "formerly-mlx_localize" logic into a stable
API surface so host apps can avoid re-implementing difference-image centroiding,
PRF-lite hypothesis scoring, baseline-sensitivity checks, and verdict mapping.

All functions are compute-only (no I/O, no network).
"""

from __future__ import annotations

import time as _time
from typing import Any, Literal, TypedDict

import numpy as np

from bittr_tess_vetter.api.pixel_prf import (
    MARGIN_RESOLVE_THRESHOLD,
    PRFParams,
    aggregate_multi_sector,
    prf_params_from_dict,
    score_hypotheses_prf_lite,
    score_hypotheses_with_prf,
)
from bittr_tess_vetter.api.references import (
    BRYSON_2010,
    BRYSON_2013,
    CALABRETTA_GREISEN_2002,
    GREISEN_CALABRETTA_2002,
    TWICKEN_2018,
    cite,
    cites,
)
from bittr_tess_vetter.api.wcs_localization import compute_difference_image_centroid_diagnostics
from bittr_tess_vetter.api.wcs_utils import compute_pixel_scale, pixel_to_world, world_to_pixel
from bittr_tess_vetter.pixel.tpf_fits import TPFFitsData

ACTION_HINT_REVIEW_MARGIN_THRESHOLD = 10.0


class ReferenceSource(TypedDict, total=False):
    name: str
    source_id: str
    ra: float
    dec: float
    # Optional pixel coordinates for WCS-less/testing flows.
    row: float
    col: float


class BaselineConsistencyResult(TypedDict):
    checked: bool
    centroid_shift_pixels: float | None
    centroid_shift_threshold_pixels: float
    verdict_local: str | None
    verdict_global: str | None
    inconsistent: bool | None


class PixelLocalizeSectorResult(TypedDict, total=False):
    status: str
    verdict: str
    raw_verdict: str

    best_source_id: str | None
    best_source_name: str | None
    margin: float | None

    centroid_row: float | None
    centroid_col: float | None
    centroid_ra_deg: float | None
    centroid_dec_deg: float | None

    sigma_row_pix: float | None
    sigma_col_pix: float | None
    sigma_arcsec: float | None

    warnings: list[str]
    hypotheses_ranked: list[dict[str, Any]]
    n_in_transit: int
    n_out_of_transit: int
    runtime_seconds: float

    prf_backend: str
    prf_fit_diagnostics: dict[str, Any] | None
    reliability_flagged: bool
    reliability_flags: list[str]
    interpretation_code: str | None
    ranking_changed_by_prior: bool
    brightness_prior_enabled: bool

    diagnostics: dict[str, Any]
    baseline_consistency: BaselineConsistencyResult
    baseline_global: dict[str, Any] | None


class PixelLocalizeMultiSectorResult(TypedDict):
    per_sector_results: list[PixelLocalizeSectorResult]
    consensus: dict[str, Any]


def _is_target_best(
    *,
    best_source_id: str | None,
    best_source_name: str | None,
    tpf_fits: TPFFitsData,
) -> bool:
    if best_source_id is None and best_source_name is None:
        return False
    target_ids = {"target", f"tic:{int(getattr(tpf_fits.ref, 'tic_id', -1))}"}
    if best_source_id in target_ids:
        return True
    return bool(best_source_name and "target" in best_source_name.lower())


def _cadence_label(cadence_seconds: float) -> str:
    if not np.isfinite(cadence_seconds) or cadence_seconds <= 0:
        return "unknown"
    if abs(cadence_seconds - 20.0) / 20.0 < 0.2:
        return "20s"
    if abs(cadence_seconds - 120.0) / 120.0 < 0.2:
        return "120s"
    return "unknown"


def _compute_tpf_cadence_summary(
    tpf_fits: TPFFitsData,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    time = np.asarray(getattr(tpf_fits, "time", []), dtype=np.float64)
    finite = np.isfinite(time)
    dt_days = float("nan")
    if int(np.sum(finite)) >= 3:
        diffs = np.diff(time[finite])
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size:
            dt_days = float(np.nanmedian(diffs))
    cadence_days = float(dt_days)
    cadence_seconds = float(dt_days * 86400.0) if np.isfinite(dt_days) else float("nan")

    n_total = int(time.shape[0])
    n_used = int(diagnostics.get("n_cadences_used", n_total)) if diagnostics else n_total
    n_dropped = (
        int(diagnostics.get("n_cadences_dropped", max(0, n_total - n_used)))
        if diagnostics
        else max(0, n_total - n_used)
    )
    dropped_fraction = float(n_dropped) / float(n_total) if n_total > 0 else float("nan")

    return {
        "n_cadences_total": n_total,
        "n_cadences_used": n_used,
        "n_cadences_dropped": n_dropped,
        "dropped_fraction": dropped_fraction,
        "cadence_seconds": cadence_seconds,
        "cadence_days": cadence_days,
        "cadence_label": _cadence_label(cadence_seconds),
    }


_NEGATIVE_NON_PHYSICAL_KEYS = {
    "fit_amplitude",
    "amplitude",
    "flux_contribution",
    "fit_flux",
    "fitted_flux",
    "host_flux_contribution",
}
_BOOLEAN_NON_PHYSICAL_KEYS = {
    "non_physical",
    "non_physical_fit",
    "is_non_physical",
    "negative_amplitude",
    "negative_flux_contribution",
}


def _collect_non_physical_prf_indicators(
    payload: Any,
    *,
    path: str = "",
    out: list[str] | None = None,
) -> list[str]:
    if out is None:
        out = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_str = str(key)
            key_l = key_str.lower()
            key_path = f"{path}.{key_str}" if path else key_str
            if key_l in _BOOLEAN_NON_PHYSICAL_KEYS and bool(value):
                out.append(f"{key_path}=true")
            if key_l in _NEGATIVE_NON_PHYSICAL_KEYS:
                try:
                    val = float(value)
                    if np.isfinite(val) and val < 0.0:
                        out.append(f"{key_path}={val:.6g}")
                except Exception:
                    pass
            if isinstance(value, (dict, list, tuple)):
                _collect_non_physical_prf_indicators(value, path=key_path, out=out)
        return out
    if isinstance(payload, (list, tuple)):
        for idx, value in enumerate(payload):
            idx_path = f"{path}[{idx}]"
            if isinstance(value, (dict, list, tuple)):
                _collect_non_physical_prf_indicators(value, path=idx_path, out=out)
        return out
    return out


def _extract_delta_mag_for_source(
    *,
    source: ReferenceSource,
    source_id: str,
    target_source_id: str,
    target_g_mag: float | None,
) -> float | None:
    if source_id == target_source_id or source_id == "target":
        return 0.0
    direct_delta = source.get("delta_mag")
    try:
        if direct_delta is not None:
            out = float(direct_delta)
            if np.isfinite(out):
                return out
    except Exception:
        pass

    meta = source.get("meta")
    if isinstance(meta, dict):
        meta_delta = meta.get("delta_mag")
        try:
            if meta_delta is not None:
                out = float(meta_delta)
                if np.isfinite(out):
                    return out
        except Exception:
            pass

    g_mag = source.get("g_mag")
    try:
        if g_mag is not None and target_g_mag is not None:
            out = float(g_mag) - float(target_g_mag)
            if np.isfinite(out):
                return out
    except Exception:
        pass
    return None


def _apply_brightness_prior(
    *,
    ranked: list[dict[str, Any]],
    reference_sources: list[ReferenceSource],
    target_source_id: str,
    weight: float,
    softening_delta_mag: float,
) -> tuple[list[dict[str, Any]], bool]:
    by_source_id: dict[str, ReferenceSource] = {}
    target_g_mag: float | None = None
    for src in reference_sources:
        sid = str(src.get("source_id") or src.get("name") or "").strip()
        if sid:
            by_source_id[sid] = src
            if sid == target_source_id:
                try:
                    g_val = src.get("g_mag")
                    if g_val is not None:
                        g_f = float(g_val)
                        if np.isfinite(g_f):
                            target_g_mag = g_f
                except Exception:
                    pass

    scored_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(ranked):
        src_id = str(row.get("source_id") or "")
        src = by_source_id.get(src_id, {})
        delta_mag = _extract_delta_mag_for_source(
            source=src,
            source_id=src_id,
            target_source_id=target_source_id,
            target_g_mag=target_g_mag,
        )
        fit_loss_raw = row.get("fit_loss")
        try:
            fit_loss_f = float(fit_loss_raw)
            if not np.isfinite(fit_loss_f):
                fit_loss_f = float("inf")
        except Exception:
            fit_loss_f = float("inf")
        prior_penalty = 0.0
        if (
            src_id not in {target_source_id, "target"}
            and delta_mag is not None
            and np.isfinite(delta_mag)
            and delta_mag > float(softening_delta_mag)
        ):
            prior_penalty = float(weight) * float(delta_mag - float(softening_delta_mag)) ** 2
        combined_loss = fit_loss_f + prior_penalty
        scored_rows.append(
            {
                "row": dict(row),
                "src_id": src_id,
                "fit_loss_raw": fit_loss_f,
                "prior_penalty": float(prior_penalty),
                "combined_loss": float(combined_loss),
                "delta_mag": float(delta_mag) if delta_mag is not None and np.isfinite(delta_mag) else None,
                "rank_raw": int(idx + 1),
                "delta_loss_raw": row.get("delta_loss"),
            }
        )

    if not scored_rows:
        return ranked, False

    raw_best_source_id = str(ranked[0].get("source_id") or "")
    scored_rows.sort(key=lambda r: (r["combined_loss"], r["fit_loss_raw"], r["rank_raw"]))
    adjusted_best_loss = float(scored_rows[0]["combined_loss"])
    adjusted: list[dict[str, Any]] = []
    for rank_idx, item in enumerate(scored_rows, start=1):
        row = dict(item["row"])
        combined_loss = float(item["combined_loss"])
        row["fit_loss_raw"] = float(item["fit_loss_raw"])
        row["brightness_prior_penalty"] = float(item["prior_penalty"])
        row["combined_loss"] = combined_loss
        row["brightness_delta_mag"] = item["delta_mag"]
        row["rank_raw"] = int(item["rank_raw"])
        row["delta_loss_raw"] = item["delta_loss_raw"]
        row["rank"] = int(rank_idx)
        row["delta_loss"] = (
            float(combined_loss - adjusted_best_loss)
            if np.isfinite(combined_loss) and np.isfinite(adjusted_best_loss)
            else 0.0
        )
        adjusted.append(row)

    ranking_changed = raw_best_source_id != str(adjusted[0].get("source_id") or "")
    return adjusted, ranking_changed


def _summarize_prior_effect(
    *,
    n_sectors_total: int,
    n_sectors_changed: int,
) -> str:
    if n_sectors_total <= 0 or n_sectors_changed <= 0:
        return "none"
    frac_changed = float(n_sectors_changed) / float(n_sectors_total)
    if n_sectors_changed >= 2 or frac_changed >= 0.5:
        return "major"
    return "minor"


def _derive_action_hint(
    *,
    consensus_best_source_id: str | None,
    consensus_margin: float | None,
    reliability_flagged: bool,
    interpretation_code: str | None,
    target_source_id: str,
) -> str:
    if reliability_flagged or interpretation_code == "INSUFFICIENT_DISCRIMINATION":
        return "DEFER_HOST_ASSIGNMENT"
    if consensus_margin is None or float(consensus_margin) < float(
        ACTION_HINT_REVIEW_MARGIN_THRESHOLD
    ):
        return "REVIEW_WITH_DILUTION"
    if consensus_best_source_id in {target_source_id, "target"}:
        return "HOST_ON_TARGET_SUPPORTED"
    return "HOST_OFF_TARGET_CANDIDATE_REVIEW"


def _annotate_fit_physical(hypotheses: list[dict[str, Any]]) -> None:
    for row in hypotheses:
        if "fit_amplitude" not in row:
            continue
        try:
            fit_amplitude = float(row.get("fit_amplitude"))
        except Exception:
            continue
        if np.isfinite(fit_amplitude):
            row["fit_physical"] = bool(fit_amplitude >= 0.0)


@cites(
    cite(BRYSON_2013, "difference-image centroid offsets and localization diagnostics"),
    cite(TWICKEN_2018, "difference image centroiding / DV-like diagnostics"),
    cite(BRYSON_2010, "PRF-based hypothesis scoring"),
    cite(GREISEN_CALABRETTA_2002, "FITS WCS framework (pixel↔sky transforms)"),
    cite(CALABRETTA_GREISEN_2002, "celestial WCS conventions"),
)
def localize_transit_host_single_sector(
    *,
    tpf_fits: TPFFitsData,
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    reference_sources: list[ReferenceSource],
    oot_margin_mult: float = 1.5,
    oot_window_mult: float | None = 10.0,
    centroid_method: Literal["centroid", "gaussian_fit"] = "centroid",
    prf_backend: Literal["prf_lite", "parametric", "instrument"] = "prf_lite",
    prf_params: dict[str, Any] | None = None,
    random_seed: int = 42,
    brightness_prior_enabled: bool = True,
    brightness_prior_weight: float = 40.0,
    brightness_prior_softening_mag: float = 2.5,
) -> PixelLocalizeSectorResult:
    start = _time.perf_counter()

    pixel_scale_arcsec = (
        float(compute_pixel_scale(tpf_fits.wcs)) if getattr(tpf_fits, "wcs", None) else 21.0
    )

    prf_backend_used: str = prf_backend
    prf_params_obj: PRFParams | None = None
    if prf_backend != "prf_lite":
        try:
            prf_params_obj = prf_params_from_dict(prf_params or {})
        except Exception:
            prf_params_obj = None
            prf_backend_used = "prf_lite"

    try:
        centroid_rc, diff_image, diff_diag = compute_difference_image_centroid_diagnostics(
            tpf_fits=tpf_fits,
            period=period_days,
            t0=t0_btjd,
            duration_hours=duration_hours,
            oot_margin_mult=oot_margin_mult,
            oot_window_mult=oot_window_mult,
            method=centroid_method,
        )
        n_in = int(diff_diag.get("n_in_transit", 0))
        n_out = int(diff_diag.get("n_out_of_transit", 0))
    except Exception as e:
        return PixelLocalizeSectorResult(
            status="invalid",
            verdict="INVALID",
            best_source_id=None,
            best_source_name=None,
            margin=None,
            centroid_row=None,
            centroid_col=None,
            centroid_ra_deg=None,
            centroid_dec_deg=None,
            sigma_row_pix=None,
            sigma_col_pix=None,
            sigma_arcsec=None,
            warnings=[f"Difference image computation failed: {type(e).__name__}: {e}"],
            hypotheses_ranked=[],
            n_in_transit=0,
            n_out_of_transit=0,
            runtime_seconds=float(_time.perf_counter() - start),
            prf_backend=prf_backend_used,
            prf_fit_diagnostics=None,
            diagnostics={},
        )

    if n_in < 1 or n_out < 1:
        return PixelLocalizeSectorResult(
            status="invalid",
            verdict="INVALID",
            best_source_id=None,
            best_source_name=None,
            margin=None,
            centroid_row=None,
            centroid_col=None,
            centroid_ra_deg=None,
            centroid_dec_deg=None,
            sigma_row_pix=None,
            sigma_col_pix=None,
            sigma_arcsec=None,
            warnings=[f"Insufficient in-transit ({n_in}) or out-of-transit ({n_out}) data"],
            hypotheses_ranked=[],
            n_in_transit=n_in,
            n_out_of_transit=n_out,
            runtime_seconds=float(_time.perf_counter() - start),
            prf_backend=prf_backend_used,
            prf_fit_diagnostics=None,
            diagnostics=dict(diff_diag),
        )

    # Convert centroid to sky coordinates
    try:
        centroid_ra, centroid_dec = pixel_to_world(
            tpf_fits.wcs, float(centroid_rc[0]), float(centroid_rc[1])
        )
    except Exception:
        centroid_ra, centroid_dec = float("nan"), float("nan")

    # Build hypotheses list for scoring.
    hyp_for_scoring: list[dict[str, Any]] = []
    for src in reference_sources:
        sid = str(src.get("source_id") or src.get("name") or "unknown")
        sname = str(src.get("name") or sid)
        row = src.get("row")
        col = src.get("col")
        if row is None or col is None:
            ra = src.get("ra")
            dec = src.get("dec")
            if ra is None or dec is None or getattr(tpf_fits, "wcs", None) is None:
                continue
            try:
                row, col = world_to_pixel(tpf_fits.wcs, float(ra), float(dec))
            except Exception:
                continue
        hyp_for_scoring.append(
            {
                "source_id": sid,
                "source_name": sname,
                "row": float(row),
                "col": float(col),
            }
        )

    ranked: list[dict[str, Any]]
    if len(hyp_for_scoring) == 0:
        ranked = []
    elif prf_backend_used == "prf_lite":
        ranked = list(
            score_hypotheses_prf_lite(
                diff_image.astype(np.float64), hyp_for_scoring, seed=random_seed
            )
        )
    else:
        ranked = list(
            score_hypotheses_with_prf(
                diff_image.astype(np.float64),
                hyp_for_scoring,
                prf_backend=prf_backend_used,  # type: ignore[arg-type]
                prf_params=prf_params_obj,
                fit_background=True,
                seed=random_seed,
            )
        )

    ranking_changed_by_prior = False
    if ranked and bool(brightness_prior_enabled):
        ranked, ranking_changed_by_prior = _apply_brightness_prior(
            ranked=ranked,
            reference_sources=reference_sources,
            target_source_id=f"tic:{int(getattr(tpf_fits.ref, 'tic_id', -1))}",
            weight=float(brightness_prior_weight),
            softening_delta_mag=float(brightness_prior_softening_mag),
        )
    if ranked:
        _annotate_fit_physical(ranked)

    warnings: list[str] = []
    reliability_flags: list[str] = []
    best_source_id: str | None = None
    best_source_name: str | None = None
    margin: float | None = None
    verdict = "AMBIGUOUS"
    raw_verdict = verdict

    if ranked:
        best_source_id = ranked[0].get("source_id")
        best_source_name = ranked[0].get("source_name")

        if len(ranked) >= 2:
            margin = ranked[1].get("delta_loss", 0.0)
        else:
            margin = None
            warnings.append(
                "Only one hypothesis provided; cannot compute margin-based host preference."
            )

        if margin is not None and float(margin) >= float(MARGIN_RESOLVE_THRESHOLD):
            verdict = (
                "ON_TARGET"
                if _is_target_best(
                    best_source_id=best_source_id,
                    best_source_name=best_source_name,
                    tpf_fits=tpf_fits,
                )
                else "OFF_TARGET"
            )
        else:
            verdict = "AMBIGUOUS"
            if margin is not None and float(margin) < float(MARGIN_RESOLVE_THRESHOLD):
                warnings.append(f"Low margin ({float(margin):.3f}) between hypotheses")
        raw_verdict = verdict

    # Uncertainty estimate (kept intentionally simple for parity)
    sigma_pix = 1.0
    sigma_arcsec = sigma_pix * float(pixel_scale_arcsec)

    prf_fit_diagnostics: dict[str, Any] | None = None
    non_physical_reasons: list[str] = []
    if ranked:
        best_hyp = ranked[0]
        non_physical_reasons = _collect_non_physical_prf_indicators(best_hyp)
        prf_fit_diagnostics = {"prf_backend": prf_backend_used}
        if prf_backend_used != "prf_lite":
            prf_fit_diagnostics.update(
                {
                    "best_log_likelihood": best_hyp.get("log_likelihood"),
                    "best_fit_residual_rms": best_hyp.get("fit_residual_rms"),
                    "best_fitted_background": best_hyp.get("fitted_background"),
                }
            )
        if non_physical_reasons:
            prf_fit_diagnostics["non_physical_indicators"] = list(non_physical_reasons)

    if non_physical_reasons:
        reliability_flags.append("NON_PHYSICAL_PRF_BEST_FIT")
        if verdict != "AMBIGUOUS":
            # Preserve pre-gate decision for auditability.
            raw_verdict = verdict
            verdict = "AMBIGUOUS"
            warnings.append(
                "Non-physical PRF best-fit indicators detected; downgraded verdict to AMBIGUOUS."
            )
        else:
            warnings.append("Non-physical PRF best-fit indicators detected.")

    reliability_flagged = bool(reliability_flags)
    interpretation_code: str | None = None
    low_margin = margin is None or float(margin) < float(MARGIN_RESOLVE_THRESHOLD)
    if low_margin or reliability_flagged:
        interpretation_code = "INSUFFICIENT_DISCRIMINATION"

    end = _time.perf_counter()
    return PixelLocalizeSectorResult(
        status="ok",
        verdict=verdict,
        raw_verdict=raw_verdict,
        best_source_id=best_source_id,
        best_source_name=best_source_name,
        margin=margin,
        centroid_row=float(centroid_rc[0]),
        centroid_col=float(centroid_rc[1]),
        centroid_ra_deg=float(centroid_ra),
        centroid_dec_deg=float(centroid_dec),
        sigma_row_pix=float(sigma_pix),
        sigma_col_pix=float(sigma_pix),
        sigma_arcsec=float(sigma_arcsec),
        warnings=warnings,
        hypotheses_ranked=ranked,
        n_in_transit=int(n_in),
        n_out_of_transit=int(n_out),
        runtime_seconds=float(end - start),
        prf_backend=prf_backend_used,
        prf_fit_diagnostics=prf_fit_diagnostics,
        reliability_flagged=reliability_flagged,
        reliability_flags=reliability_flags,
        interpretation_code=interpretation_code,
        ranking_changed_by_prior=bool(ranking_changed_by_prior),
        brightness_prior_enabled=bool(brightness_prior_enabled),
        diagnostics=dict(diff_diag),
    )


@cites(
    cite(BRYSON_2013, "baseline sensitivity as a localization robustness check"),
    cite(TWICKEN_2018, "difference-image centroid consistency diagnostics"),
)
def localize_transit_host_single_sector_with_baseline_check(
    *,
    tpf_fits: TPFFitsData,
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    reference_sources: list[ReferenceSource],
    oot_margin_mult: float = 1.5,
    oot_window_mult: float | None = 10.0,
    centroid_method: Literal["centroid", "gaussian_fit"] = "centroid",
    prf_backend: Literal["prf_lite", "parametric", "instrument"] = "prf_lite",
    prf_params: dict[str, Any] | None = None,
    random_seed: int = 42,
    centroid_shift_threshold_pixels: float = 0.5,
    brightness_prior_enabled: bool = True,
    brightness_prior_weight: float = 40.0,
    brightness_prior_softening_mag: float = 2.5,
) -> PixelLocalizeSectorResult:
    local = localize_transit_host_single_sector(
        tpf_fits=tpf_fits,
        period_days=period_days,
        t0_btjd=t0_btjd,
        duration_hours=duration_hours,
        reference_sources=reference_sources,
        oot_margin_mult=oot_margin_mult,
        oot_window_mult=oot_window_mult,
        centroid_method=centroid_method,
        prf_backend=prf_backend,
        prf_params=prf_params,
        random_seed=random_seed,
        brightness_prior_enabled=brightness_prior_enabled,
        brightness_prior_weight=brightness_prior_weight,
        brightness_prior_softening_mag=brightness_prior_softening_mag,
    )

    checked = oot_window_mult is not None and local.get("status") == "ok"
    if not checked:
        local["baseline_consistency"] = BaselineConsistencyResult(
            checked=False,
            centroid_shift_pixels=None,
            centroid_shift_threshold_pixels=float(centroid_shift_threshold_pixels),
            verdict_local=None,
            verdict_global=None,
            inconsistent=None,
        )
        return local

    global_res = localize_transit_host_single_sector(
        tpf_fits=tpf_fits,
        period_days=period_days,
        t0_btjd=t0_btjd,
        duration_hours=duration_hours,
        reference_sources=reference_sources,
        oot_margin_mult=oot_margin_mult,
        oot_window_mult=None,
        centroid_method=centroid_method,
        prf_backend=prf_backend,
        prf_params=prf_params,
        random_seed=random_seed,
        brightness_prior_enabled=brightness_prior_enabled,
        brightness_prior_weight=brightness_prior_weight,
        brightness_prior_softening_mag=brightness_prior_softening_mag,
    )

    if global_res.get("status") != "ok":
        local["baseline_consistency"] = BaselineConsistencyResult(
            checked=False,
            centroid_shift_pixels=None,
            centroid_shift_threshold_pixels=float(centroid_shift_threshold_pixels),
            verdict_local=str(local.get("verdict") or "INVALID"),
            verdict_global=str(global_res.get("verdict") or "INVALID"),
            inconsistent=None,
        )
        return local

    rc_local = (float(local.get("centroid_row")), float(local.get("centroid_col")))
    rc_global = (float(global_res.get("centroid_row")), float(global_res.get("centroid_col")))
    shift_px = float(np.hypot(rc_local[0] - rc_global[0], rc_local[1] - rc_global[1]))
    verdict_local = str(local.get("verdict") or "INVALID")
    verdict_global = str(global_res.get("verdict") or "INVALID")
    inconsistent = bool(
        (verdict_local != verdict_global)
        or (np.isfinite(shift_px) and shift_px > float(centroid_shift_threshold_pixels))
    )

    local["baseline_consistency"] = BaselineConsistencyResult(
        checked=True,
        centroid_shift_pixels=shift_px,
        centroid_shift_threshold_pixels=float(centroid_shift_threshold_pixels),
        verdict_local=verdict_local,
        verdict_global=verdict_global,
        inconsistent=inconsistent,
    )
    local["baseline_global"] = {
        "status": global_res.get("status"),
        "verdict": verdict_global,
        "best_source_id": global_res.get("best_source_id"),
        "best_source_name": global_res.get("best_source_name"),
        "margin": global_res.get("margin"),
        "centroid_row": global_res.get("centroid_row"),
        "centroid_col": global_res.get("centroid_col"),
        "centroid_ra_deg": global_res.get("centroid_ra_deg"),
        "centroid_dec_deg": global_res.get("centroid_dec_deg"),
        "sigma_arcsec": global_res.get("sigma_arcsec"),
        "warnings": global_res.get("warnings", []),
    }

    if verdict_local == "OFF_TARGET" and inconsistent:
        local["raw_verdict"] = verdict_local
        local["verdict"] = "AMBIGUOUS"
        flags = list(local.get("reliability_flags") or [])
        if "BASELINE_SENSITIVE_LOCALIZATION" not in flags:
            flags.append("BASELINE_SENSITIVE_LOCALIZATION")
        local["reliability_flags"] = flags
        local["reliability_flagged"] = True
        local["interpretation_code"] = "INSUFFICIENT_DISCRIMINATION"
        local.setdefault("warnings", []).append(
            "Downgraded OFF_TARGET to AMBIGUOUS due to baseline-sensitive localization."
        )
    return local


@cites(
    cite(BRYSON_2013, "difference-image localization diagnostics across epochs"),
    cite(TWICKEN_2018, "DV-like multi-epoch consistency checks"),
    cite(BRYSON_2010, "PRF-based host hypothesis scoring"),
    cite(GREISEN_CALABRETTA_2002, "FITS WCS framework (pixel↔sky transforms)"),
    cite(CALABRETTA_GREISEN_2002, "celestial WCS conventions"),
)
def localize_transit_host_multi_sector(
    *,
    tpf_fits_list: list[TPFFitsData],
    period_days: float,
    t0_btjd: float,
    duration_hours: float,
    reference_sources: list[ReferenceSource],
    oot_margin_mult: float = 1.5,
    oot_window_mult: float | None = 10.0,
    centroid_method: Literal["centroid", "gaussian_fit"] = "centroid",
    prf_backend: Literal["prf_lite", "parametric", "instrument"] = "prf_lite",
    prf_params: dict[str, Any] | None = None,
    random_seed: int = 42,
    centroid_shift_threshold_pixels: float = 0.5,
    brightness_prior_enabled: bool = True,
    brightness_prior_weight: float = 40.0,
    brightness_prior_softening_mag: float = 2.5,
) -> PixelLocalizeMultiSectorResult:
    """Localize the transit host across multiple sectors and build a consensus."""
    per_sector_results: list[PixelLocalizeSectorResult] = []

    for tpf in tpf_fits_list:
        r = localize_transit_host_single_sector_with_baseline_check(
            tpf_fits=tpf,
            period_days=period_days,
            t0_btjd=t0_btjd,
            duration_hours=duration_hours,
            reference_sources=reference_sources,
            oot_margin_mult=oot_margin_mult,
            oot_window_mult=oot_window_mult,
            centroid_method=centroid_method,
            prf_backend=prf_backend,
            prf_params=prf_params,
            random_seed=random_seed,
            centroid_shift_threshold_pixels=centroid_shift_threshold_pixels,
            brightness_prior_enabled=brightness_prior_enabled,
            brightness_prior_weight=brightness_prior_weight,
            brightness_prior_softening_mag=brightness_prior_softening_mag,
        )
        # Attach lightweight metadata so callers don't need to loop just to label sectors.
        try:
            r["sector"] = int(getattr(tpf.ref, "sector", 0))
            r["tpf_fits_ref"] = str(tpf.ref.to_string())
        except Exception:
            pass
        cadence_summary = _compute_tpf_cadence_summary(tpf, r.get("diagnostics"))
        r["cadence_summary"] = cadence_summary
        dropped_fraction_raw = cadence_summary.get("dropped_fraction")
        dropped_fraction = (
            float(dropped_fraction_raw)
            if isinstance(dropped_fraction_raw, (int, float))
            else float("nan")
        )
        if np.isfinite(dropped_fraction) and dropped_fraction > 0.2:
            warnings = list(r.get("warnings") or [])
            warnings.append(
                "High cadence dropout; localization reliability reduced "
                f"(dropped_fraction={dropped_fraction:.3f})."
            )
            r["warnings"] = warnings
            flags = list(r.get("reliability_flags") or [])
            if "HIGH_CADENCE_DROPOUT" not in flags:
                flags.append("HIGH_CADENCE_DROPOUT")
            r["reliability_flags"] = flags
            r["reliability_flagged"] = True
            r["interpretation_code"] = "INSUFFICIENT_DISCRIMINATION"
        per_sector_results.append(r)

    consensus = aggregate_multi_sector(per_sector_results)
    consensus_reliability_flags = sorted(
        {
            str(flag)
            for r in per_sector_results
            for flag in (r.get("reliability_flags") or [])
            if flag is not None
        }
    )
    consensus_reliability_flagged = bool(consensus_reliability_flags)
    prior_changed_count = sum(1 for r in per_sector_results if bool(r.get("ranking_changed_by_prior")))
    consensus_margin_raw = consensus.get("consensus_margin")
    consensus_margin = (
        float(consensus_margin_raw)
        if isinstance(consensus_margin_raw, (int, float)) and np.isfinite(consensus_margin_raw)
        else None
    )
    low_discrimination = (
        consensus_margin is None or consensus_margin < float(MARGIN_RESOLVE_THRESHOLD)
    )
    if low_discrimination or consensus_reliability_flagged:
        consensus["interpretation_code"] = "INSUFFICIENT_DISCRIMINATION"
    else:
        consensus["interpretation_code"] = None
    consensus["reliability_flagged"] = consensus_reliability_flagged
    consensus["brightness_prior_enabled"] = bool(brightness_prior_enabled)
    consensus["ranking_changed_by_prior"] = prior_changed_count > 0
    consensus["n_sectors_ranking_changed_by_prior"] = int(prior_changed_count)
    consensus["prior_effect"] = _summarize_prior_effect(
        n_sectors_total=len(per_sector_results),
        n_sectors_changed=int(prior_changed_count),
    )
    target_source_id = (
        f"tic:{int(getattr(tpf_fits_list[0].ref, 'tic_id', -1))}" if len(tpf_fits_list) > 0 else "target"
    )
    consensus["action_hint"] = _derive_action_hint(
        consensus_best_source_id=str(consensus.get("consensus_best_source_id"))
        if consensus.get("consensus_best_source_id") is not None
        else None,
        consensus_margin=consensus_margin,
        reliability_flagged=consensus_reliability_flagged,
        interpretation_code=str(consensus.get("interpretation_code"))
        if consensus.get("interpretation_code") is not None
        else None,
        target_source_id=target_source_id,
    )
    if consensus_reliability_flags:
        consensus["reliability_flags"] = consensus_reliability_flags
    return PixelLocalizeMultiSectorResult(
        per_sector_results=per_sector_results, consensus=dict(consensus)
    )


__all__ = [
    "ReferenceSource",
    "BaselineConsistencyResult",
    "PixelLocalizeSectorResult",
    "PixelLocalizeMultiSectorResult",
    "localize_transit_host_single_sector",
    "localize_transit_host_single_sector_with_baseline_check",
    "localize_transit_host_multi_sector",
]
