"""Catalog-based vetting computations (metrics-only).

V06: Nearby eclipsing binary search (TESS-EB via VizieR TAP)
V07: ExoFOP TOI row lookup (downloaded table)

All results are metrics-only: `passed=None` and `details["_metrics_only"]=True`.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from typing import Any
from xml.etree import ElementTree as ET

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


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1]


def _parse_vizier_votable(text: str) -> list[dict[str, str]]:
    root = ET.fromstring(text)

    fields: list[str] = []
    for elem in root.iter():
        if _strip_ns(elem.tag) != "FIELD":
            continue
        name = elem.attrib.get("name") or elem.attrib.get("ID")
        if name:
            fields.append(name)

    rows: list[dict[str, str]] = []
    for tr in root.iter():
        if _strip_ns(tr.tag) != "TR":
            continue
        values: list[str] = []
        for td in tr:
            if _strip_ns(td.tag) != "TD":
                continue
            values.append(td.text.strip() if td.text else "")

        if not values:
            continue

        if fields and len(fields) == len(values):
            rows.append({fields[i]: values[i] for i in range(len(values))})
        else:
            # Fallback: assume the requested output column order.
            row: dict[str, str] = {}
            if len(values) >= 1:
                row["TIC"] = values[0]
            if len(values) >= 2:
                row["Per"] = values[1]
            if len(values) >= 3:
                row["RAJ2000"] = values[2]
            if len(values) >= 4:
                row["DEJ2000"] = values[3]
            rows.append(row)

    return rows


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
            details={
                "status": "error",
                "error": str(e),
                "ra": ra_deg,
                "dec": dec_deg,
            },
        )

    text = resp.text
    parsed_rows: list[dict[str, str]] = []
    try:
        parsed_rows = _parse_vizier_votable(text)
    except Exception:
        parsed_rows = []

    if not parsed_rows and "<TR>" not in text:
        plot_data = {
            "version": 1,
            "target_ra": float(ra_deg),
            "target_dec": float(dec_deg),
            "search_radius_arcsec": float(search_radius_arcsec),
            "matches": [],
        }
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
                "plot_data": plot_data,
            },
        )

    # Extremely lightweight fallback parse (VOTable-ish HTML). Extract rows by naive split.
    if not parsed_rows:
        for tr in text.split("<TR>")[1:]:
            tds = tr.split("<TD>")[1:]
            if len(tds) < 4:
                continue
            parsed_rows.append(
                {
                    "TIC": tds[0].split("</TD>")[0].strip(),
                    "Per": tds[1].split("</TD>")[0].strip(),
                    "RAJ2000": tds[2].split("</TD>")[0].strip(),
                    "DEJ2000": tds[3].split("</TD>")[0].strip(),
                }
            )

    rows: list[dict[str, Any]] = []
    for r in parsed_rows:
        try:
            tic = int(str(r.get("TIC", "")).strip())
        except Exception:
            continue
        try:
            per = float(str(r.get("Per", "")).strip())
        except Exception:
            per = float("nan")
        try:
            ra = float(str(r.get("RAJ2000", "")).strip())
        except Exception:
            ra = float("nan")
        try:
            dec = float(str(r.get("DEJ2000", "")).strip())
        except Exception:
            dec = float("nan")

        rows.append({"tic_id": tic, "period_days": per, "ra_deg": ra, "dec_deg": dec})

    candidate_period_days = (
        float(candidate_period_days) if candidate_period_days is not None else None
    )
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
        match_summaries.append(
            {
                "tic_id": r["tic_id"],
                "period_days": per,
                "ra_deg": (ra2 if math.isfinite(ra2) else None),
                "dec_deg": (dec2 if math.isfinite(dec2) else None),
                "sep_arcsec": sep_arcsec,
                **deltas,
            }
        )

    plot_data = {
        "version": 1,
        "target_ra": float(ra_deg),
        "target_dec": float(dec_deg),
        "search_radius_arcsec": float(search_radius_arcsec),
        "matches": [
            {
                "ra": m.get("ra_deg"),
                "dec": m.get("dec_deg"),
                "sep_arcsec": m.get("sep_arcsec"),
                "period_days": m.get("period_days"),
            }
            for m in match_summaries
        ],
    }

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
            "min_period_ratio_delta_any": (
                float(min_delta_any) if min_delta_any != float("inf") else None
            ),
            "matches": match_summaries,
            "plot_data": plot_data,
        },
    )


def run_exofop_toi_lookup(
    *,
    tic_id: int,
    toi: float | None = None,
    http_get: Callable[..., Any] | None = None,
) -> VetterCheckResult:
    """V07: Query ExoFOP's TOI table for `tic_id` (optionally TOI-filtered).

    Uses the shared `catalogs.exofop_toi_table` implementation which provides:
    - in-process caching
    - on-disk caching with TTL (persists across MCP restarts)
    - bounded retries/backoff for network fetches
    """
    try:
        from bittr_tess_vetter.platform.catalogs.exofop_toi_table import fetch_exofop_toi_table

        cache_ttl_seconds = 24 * 3600
        table = fetch_exofop_toi_table(cache_ttl_seconds=cache_ttl_seconds)
    except Exception as e:
        # Fallback: if the ExoFOP network fetch fails, try returning any disk cache
        # (even if stale) rather than hard-failing.
        try:
            from bittr_tess_vetter.platform.catalogs.exofop_toi_table import fetch_exofop_toi_table

            table = fetch_exofop_toi_table(cache_ttl_seconds=10**9)
            used_stale_cache = True
            fetch_error = str(e)
        except Exception:
            return _metrics_result(
                check_id="V07",
                name="exofop_toi_lookup",
                confidence=0.0,
                details={
                    "status": "error",
                    "error": str(e),
                    "tic_id": int(tic_id),
                    "note": "ExoFOP fetch failed and no disk cache was available",
                },
            )
    else:
        used_stale_cache = False
        fetch_error = None

    # Filter rows for this TIC (+ optional TOI).
    rows = list(table.entries_for_tic(int(tic_id)))
    selected: dict[str, str] | None = None
    if toi is not None:
        target_toi = float(toi)
        for r in rows:
            try:
                if abs(float(r.get("toi", "nan")) - target_toi) < 1e-6:
                    selected = r
                    break
            except Exception:
                continue
    else:
        selected = rows[0] if rows else None

    now = time.time()
    age_seconds = float(now - float(table.fetched_at_unix)) if table.fetched_at_unix else None
    is_stale = bool(age_seconds is not None and age_seconds > float(cache_ttl_seconds))

    if selected is None:
        plot_data: dict[str, Any] = {
            "version": 1,
            "tic_id": int(tic_id),
            "found": False,
        }
        return _metrics_result(
            check_id="V07",
            name="exofop_toi_lookup",
            confidence=0.7,
            details={
                "status": "ok",
                "tic_id": int(tic_id),
                "toi": (float(toi) if toi is not None else None),
                "found": False,
                "source": "exofop_toi_table",
                "cache_ttl_seconds": int(cache_ttl_seconds),
                "cache_age_seconds": age_seconds,
                "cache_stale": is_stale,
                "used_stale_cache": used_stale_cache,
                "fetch_error": fetch_error,
                "plot_data": plot_data,
            },
        )

    row = dict(selected)
    toi_str = row.get("toi")
    tfopwg = row.get("tfopwg_disp") or row.get("tfopwg") or row.get("tfopwg_disposition")
    planet_disp = row.get("disp") or row.get("planet_disposition") or row.get("disposition")
    comments = row.get("comments") or row.get("comment") or row.get("notes")

    plot_data = {
        "version": 1,
        "tic_id": int(tic_id),
        "found": True,
        "toi": toi_str,
        "tfopwg_disposition": tfopwg,
        "planet_disposition": planet_disp,
        "comments": comments,
    }

    return _metrics_result(
        check_id="V07",
        name="exofop_toi_lookup",
        confidence=0.8,
        details={
            "status": "ok",
            "tic_id": int(tic_id),
            "toi": (float(toi) if toi is not None else None),
            "found": True,
            "row": row,
            "source": "exofop_toi_table",
            "cache_ttl_seconds": int(cache_ttl_seconds),
            "cache_age_seconds": age_seconds,
            "cache_stale": is_stale,
            "used_stale_cache": used_stale_cache,
            "fetch_error": fetch_error,
            "plot_data": plot_data,
        },
    )


__all__ = ["run_nearby_eb_search", "run_exofop_toi_lookup"]
