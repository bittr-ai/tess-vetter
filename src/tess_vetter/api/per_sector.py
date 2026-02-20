"""Per-sector vetting convenience helpers (researcher-facing).

These helpers compose existing public APIs to run the same vetting pipeline
independently per sector/chunk. Outputs are policy-free and metrics-first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tess_vetter.api.sector_metrics import (
    SectorEphemerisMetrics,
    compute_sector_ephemeris_metrics,
)
from tess_vetter.api.types import (
    Candidate,
    LightCurve,
    StellarParams,
    TPFStamp,
    VettingBundleResult,
)
from tess_vetter.api.vet import vet_candidate


@dataclass(frozen=True)
class PerSectorVettingResult:
    """Structured output from :func:`per_sector_vet`."""

    schema_version: int
    bundles_by_sector: dict[int, VettingBundleResult]
    sector_ephemeris_metrics: list[SectorEphemerisMetrics]
    summary_records: list[dict[str, Any]]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "bundles_by_sector": {
                int(k): v.model_dump() for k, v in sorted(self.bundles_by_sector.items())
            },
            "sector_ephemeris_metrics": [m.to_dict() for m in self.sector_ephemeris_metrics],
            "summary_records": list(self.summary_records),
            "provenance": dict(self.provenance),
        }


def _as_arrays(lc_by_sector: dict[int, LightCurve]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    times: list[np.ndarray] = []
    fluxes: list[np.ndarray] = []
    flux_errs: list[np.ndarray] = []
    sectors: list[np.ndarray] = []

    for sec in sorted(lc_by_sector.keys()):
        lc = lc_by_sector[int(sec)]
        t = np.asarray(lc.time, dtype=np.float64)
        f = np.asarray(lc.flux, dtype=np.float64)
        e = (
            np.asarray(lc.flux_err, dtype=np.float64)
            if lc.flux_err is not None
            else np.zeros_like(f, dtype=np.float64)
        )
        s = np.full(len(t), int(sec), dtype=np.int32)

        times.append(t)
        fluxes.append(f)
        flux_errs.append(e)
        sectors.append(s)

    return (
        np.concatenate(times),
        np.concatenate(fluxes),
        np.concatenate(flux_errs),
        np.concatenate(sectors),
    )


def per_sector_vet(
    lc_by_sector: dict[int, LightCurve],
    candidate: Candidate,
    *,
    stellar: StellarParams | None = None,
    tpf_by_sector: dict[int, TPFStamp] | None = None,
    preset: str = "default",
    checks: list[str] | None = None,
    network: bool = False,
    ra_deg: float | None = None,
    dec_deg: float | None = None,
    tic_id: int | None = None,
    extra_context: dict[str, Any] | None = None,
) -> PerSectorVettingResult:
    """Run the vetting pipeline independently per sector.

    Args:
        lc_by_sector: Mapping from sector -> light curve.
        candidate: Transit candidate ephemeris/depth.
        stellar: Optional stellar parameters for checks that use them.
        tpf_by_sector: Optional mapping from sector -> TPF stamp. When provided,
            pixel-level checks (V08-V10) can run per sector.
        preset: Pipeline preset ("default" or "extended").
        checks: Optional explicit check IDs list.
        network: Whether to allow network access for catalog checks.
        ra_deg/dec_deg: Sky coordinates (for catalog checks).
        tic_id: TIC ID for provenance/caches.
        extra_context: Additional context dict passed through to checks.

    Returns:
        PerSectorVettingResult with per-sector bundles and a metrics-only per-sector
        ephemeris diagnostics table.
    """
    if not lc_by_sector:
        raise ValueError("lc_by_sector cannot be empty")

    bundles_by_sector: dict[int, VettingBundleResult] = {}
    summary_records: list[dict[str, Any]] = []

    for sec in sorted(lc_by_sector.keys()):
        lc = lc_by_sector[int(sec)]
        tpf = tpf_by_sector.get(int(sec)) if tpf_by_sector else None
        context = {"sector": int(sec)}
        if extra_context:
            context.update({str(k): v for k, v in extra_context.items()})

        bundle = vet_candidate(
            lc,
            candidate,
            stellar=stellar,
            tpf=tpf,
            network=network,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            tic_id=tic_id,
            preset=preset,
            checks=checks,
            context=context,
        )
        bundles_by_sector[int(sec)] = bundle
        summary_records.append(
            {
                "sector": int(sec),
                "checks": int(len(bundle.results)),
                "ok": int(bundle.n_passed),
                "error": int(bundle.n_failed),
                "skipped": int(bundle.n_unknown),
            }
        )

    time, flux, flux_err, sector = _as_arrays(lc_by_sector)
    metrics = compute_sector_ephemeris_metrics(
        time=time,
        flux=flux,
        flux_err=flux_err,
        sector=sector,
        period_days=float(candidate.ephemeris.period_days),
        t0_btjd=float(candidate.ephemeris.t0_btjd),
        duration_hours=float(candidate.ephemeris.duration_hours),
    )

    provenance = {
        "schema_version": 1,
        "preset": str(preset),
        "checks": list(checks) if checks is not None else None,
        "sectors": sorted(int(s) for s in lc_by_sector),
        "has_tpf_by_sector": bool(tpf_by_sector),
    }

    return PerSectorVettingResult(
        schema_version=1,
        bundles_by_sector=bundles_by_sector,
        sector_ephemeris_metrics=metrics,
        summary_records=summary_records,
        provenance=provenance,
    )


__all__ = ["PerSectorVettingResult", "per_sector_vet"]

