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
    prf_params_from_dict,
    score_hypotheses_prf_lite,
    score_hypotheses_with_prf,
)
from bittr_tess_vetter.api.wcs_localization import compute_difference_image_centroid_diagnostics
from bittr_tess_vetter.api.wcs_utils import compute_pixel_scale, pixel_to_world, world_to_pixel

if False:  # TYPE_CHECKING
    from bittr_tess_vetter.pixel.tpf_fits import TPFFitsData


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

    diagnostics: dict[str, Any]
    baseline_consistency: BaselineConsistencyResult
    baseline_global: dict[str, Any] | None


def _is_target_best(
    *,
    best_source_id: str | None,
    best_source_name: str | None,
    tpf_fits: "TPFFitsData",
) -> bool:
    if best_source_id is None and best_source_name is None:
        return False
    target_ids = {"target", f"tic:{int(getattr(tpf_fits.ref, 'tic_id', -1))}"}
    if best_source_id in target_ids:
        return True
    if best_source_name and "target" in best_source_name.lower():
        return True
    return False


def localize_transit_host_single_sector(
    *,
    tpf_fits: "TPFFitsData",
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
) -> PixelLocalizeSectorResult:
    start = _time.perf_counter()

    pixel_scale_arcsec = float(compute_pixel_scale(tpf_fits.wcs)) if getattr(tpf_fits, "wcs", None) else 21.0

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
        centroid_ra, centroid_dec = pixel_to_world(tpf_fits.wcs, float(centroid_rc[0]), float(centroid_rc[1]))
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
        ranked = list(score_hypotheses_prf_lite(diff_image.astype(np.float64), hyp_for_scoring, seed=random_seed))
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

    warnings: list[str] = []
    best_source_id: str | None = None
    best_source_name: str | None = None
    margin: float | None = None
    verdict = "AMBIGUOUS"

    if ranked:
        best_source_id = ranked[0].get("source_id")
        best_source_name = ranked[0].get("source_name")

        if len(ranked) >= 2:
            margin = ranked[1].get("delta_loss", 0.0)
        else:
            margin = None
            warnings.append("Only one hypothesis provided; cannot compute margin-based host preference.")

        if margin is not None and float(margin) >= float(MARGIN_RESOLVE_THRESHOLD):
            verdict = "ON_TARGET" if _is_target_best(
                best_source_id=best_source_id,
                best_source_name=best_source_name,
                tpf_fits=tpf_fits,
            ) else "OFF_TARGET"
        else:
            verdict = "AMBIGUOUS"
            if margin is not None and float(margin) < float(MARGIN_RESOLVE_THRESHOLD):
                warnings.append(f"Low margin ({float(margin):.3f}) between hypotheses")

    # Uncertainty estimate (kept intentionally simple for parity)
    sigma_pix = 1.0
    sigma_arcsec = sigma_pix * float(pixel_scale_arcsec)

    prf_fit_diagnostics: dict[str, Any] | None = None
    if prf_backend_used != "prf_lite" and ranked:
        best_hyp = ranked[0]
        prf_fit_diagnostics = {
            "prf_backend": prf_backend_used,
            "best_log_likelihood": best_hyp.get("log_likelihood"),
            "best_fit_residual_rms": best_hyp.get("fit_residual_rms"),
            "best_fitted_background": best_hyp.get("fitted_background"),
        }

    end = _time.perf_counter()
    return PixelLocalizeSectorResult(
        status="ok",
        verdict=verdict,
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
        diagnostics=dict(diff_diag),
    )


def localize_transit_host_single_sector_with_baseline_check(
    *,
    tpf_fits: "TPFFitsData",
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
        local.setdefault("warnings", []).append(
            "Downgraded OFF_TARGET to AMBIGUOUS due to baseline-sensitive localization."
        )
    return local


__all__ = [
    "ReferenceSource",
    "BaselineConsistencyResult",
    "PixelLocalizeSectorResult",
    "localize_transit_host_single_sector",
    "localize_transit_host_single_sector_with_baseline_check",
]

