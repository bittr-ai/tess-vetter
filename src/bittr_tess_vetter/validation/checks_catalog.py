"""Catalog-based vetting computations (metrics-only).

V06: Nearby eclipsing binary search (TESS-EB via VizieR TAP)
V07: ExoFOP TOI row lookup (downloaded table)

All results are metrics-only: `passed=None` and `details["_metrics_only"]=True`.
"""

from __future__ import annotations

import csv
import io
import logging
import math
from typing import Any, Callable

import requests

from bittr_tess_vetter.domain.detection import VetterCheckResult

logger = logging.getLogger(__name__)

VIZIER_TAP_URL = "https://vizier.cds.unistra.fr/viz-bin/votable"
EXOFOP_TOI_URL = "https://exofop.ipac.caltech.edu/tess/download_toi.php"
REQUEST_TIMEOUT_S = 10.0


def _metrics_result(
    *,
    check_id: str,
    name: str,
    confidence: float,
    details: dict[str, Any],
) -> VetterCheckResult:
    details = dict(details)
    details["_metrics_only"] = True
    return VetterCheckResult(
        id=check_id,
        name=name,
        passed=None,
        confidence=float(max(0.0, min(1.0, confidence))),
        details=details,
    )


def _ratio_deltas(eb_period: float, cand_period: float) -> dict[str, float]:
    if cand_period <= 0 or eb_period <= 0:
        return {"delta_1x": float("inf"), "delta_2x": float("inf"), "delta_0p5x": float("inf")}
    r = eb_period / cand_period
    return {"delta_1x": abs(r - 1.0), "delta_2x": abs(r - 2.0), "delta_0p5x": abs(r - 0.5)}


def _angular_separation_arcsec(
    ra1_deg: float,
    dec1_deg: float,
    ra2_deg: float,
    dec2_deg: float,
) -> float:
    """Spherical angular separation in arcseconds (haversine, stable for small angles)."""
    ra1 = math.radians(ra1_deg)
    dec1 = math.radians(dec1_deg)
    ra2 = math.radians(ra2_deg)
    dec2 = math.radians(dec2_deg)

    sin_ddec = math.sin((dec2 - dec1) / 2.0)
    sin_dra = math.sin((ra2 - ra1) / 2.0)
    a = sin_ddec * sin_ddec + math.cos(dec1) * math.cos(dec2) * sin_dra * sin_dra
    a = min(1.0, max(0.0, a))
    c = 2.0 * math.asin(math.sqrt(a))
    return float(math.degrees(c) * 3600.0)

def run_nearby_eb_search(
    *,
    ra_deg: float,
    dec_deg: float,
    candidate_period_days: float | None = None,
    search_radius_arcsec: float = 42.0,
    http_get: Callable[..., Any] | None = None,
) -> VetterCheckResult:
    """V06: Query the TESS-EB catalog near a position and return raw matches."""
    http_get = http_get or requests.get

    params = {
        "-source": "J/ApJS/258/16/tess-ebs",
        "-c": f"{ra_deg} {dec_deg}",
        "-c.rs": f"{search_radius_arcsec / 3600.0}",
        "-out": "TIC,Per,RAJ2000,DEJ2000",
        "-out.max": "50",
    }

    try:
        resp = http_get(VIZIER_TAP_URL, params=params, timeout=REQUEST_TIMEOUT_S)
        resp.raise_for_status()
    except Exception as e:
        return _metrics_result(
            check_id="V06",
            name="nearby_eb_search",
            confidence=0.0,
            details={"status": "error", "error": str(e), "ra": ra_deg, "dec": dec_deg},
        )

    text = resp.text
    if "<TR>" not in text:
        return _metrics_result(
            check_id="V06",
            name="nearby_eb_search",
            confidence=0.6,
            details={
                "status": "ok",
                "n_ebs_found": 0,
                "search_radius_arcsec": search_radius_arcsec,
                "ra": ra_deg,
                "dec": dec_deg,
            },
        )

    # Extremely lightweight parse (VOTable-ish HTML). Extract rows by naive split.
    rows: list[dict[str, Any]] = []
    for tr in text.split("<TR>")[1:]:
        tds = tr.split("<TD>")[1:]
        if len(tds) < 4:
            continue
        try:
            tic = int(tds[0].split("</TD>")[0].strip())
        except Exception:
            continue
        try:
            per = float(tds[1].split("</TD>")[0].strip())
        except Exception:
            per = float("nan")
        try:
            ra = float(tds[2].split("</TD>")[0].strip())
        except Exception:
            ra = float("nan")
        try:
            dec = float(tds[3].split("</TD>")[0].strip())
        except Exception:
            dec = float("nan")
        rows.append({"tic_id": tic, "period_days": per, "ra_deg": ra, "dec_deg": dec})

    candidate_period_days = float(candidate_period_days) if candidate_period_days is not None else None
    match_summaries: list[dict[str, Any]] = []
    min_delta_any = float("inf")

    for r in rows:
        per = float(r.get("period_days") or float("nan"))
        if candidate_period_days is None or not (candidate_period_days > 0) or not (per > 0):
            deltas = {"delta_1x": None, "delta_2x": None, "delta_0p5x": None}
        else:
            d = _ratio_deltas(per, candidate_period_days)
            deltas = {k: float(v) for k, v in d.items()}
            min_delta_any = min(min_delta_any, min(d.values()))
        ra2 = float(r.get("ra_deg") or float("nan"))
        dec2 = float(r.get("dec_deg") or float("nan"))
        sep_arcsec = (
            _angular_separation_arcsec(float(ra_deg), float(dec_deg), ra2, dec2)
            if (math.isfinite(ra2) and math.isfinite(dec2))
            else None
        )
        match_summaries.append({"tic_id": r["tic_id"], "period_days": per, "sep_arcsec": sep_arcsec, **deltas})

    return _metrics_result(
        check_id="V06",
        name="nearby_eb_search",
        confidence=0.8,
        details={
            "status": "ok",
            "search_radius_arcsec": float(search_radius_arcsec),
            "ra": float(ra_deg),
            "dec": float(dec_deg),
            "candidate_period_days": candidate_period_days,
            "n_ebs_found": int(len(match_summaries)),
            "min_period_ratio_delta_any": (float(min_delta_any) if min_delta_any != float("inf") else None),
            "matches": match_summaries,
        },
    )


def run_exofop_toi_lookup(
    *,
    tic_id: int,
    toi: float | None = None,
    http_get: Callable[..., Any] | None = None,
) -> VetterCheckResult:
    """V07: Download ExoFOP's TOI table and return the row for `tic_id` (optionally TOI-filtered)."""
    http_get = http_get or requests.get
    try:
        resp = http_get(EXOFOP_TOI_URL, timeout=REQUEST_TIMEOUT_S)
        resp.raise_for_status()
    except Exception as e:
        return _metrics_result(
            check_id="V07",
            name="exofop_toi_lookup",
            confidence=0.0,
            details={"status": "error", "error": str(e), "tic_id": int(tic_id)},
        )

    csv_text = resp.text
    reader = csv.DictReader(io.StringIO(csv_text))
    row: dict[str, Any] | None = None
    for r in reader:
        try:
            if int(r.get("TIC ID", "0")) != int(tic_id):
                continue
            if toi is not None:
                toi_val = float(r.get("TOI", "nan"))
                if not (abs(toi_val - float(toi)) < 1e-6):
                    continue
            row = r
            break
        except Exception:
            continue

    if row is None:
        return _metrics_result(
            check_id="V07",
            name="exofop_toi_lookup",
            confidence=0.7,
            details={"status": "ok", "tic_id": int(tic_id), "found": False},
        )

    # Keep raw strings; downstream code can interpret dispositions.
    return _metrics_result(
        check_id="V07",
        name="exofop_toi_lookup",
        confidence=0.8,
        details={"status": "ok", "tic_id": int(tic_id), "found": True, "row": dict(row)},
    )


__all__ = ["run_nearby_eb_search", "run_exofop_toi_lookup"]
