from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from tess_vetter.features import FeatureConfig
from tess_vetter.pipeline import enrich_candidate


@dataclass(frozen=True)
class _FakeSearchRow:
    sector: int
    exptime: float = 120.0


@dataclass
class _FakeBundle:
    results: list[object]
    warnings: list[str]
    inputs_summary: dict[str, object]
    provenance: dict[str, object]


class _FakeCheck:
    def __init__(self, check_id: str, metrics: dict[str, object] | None = None) -> None:
        self._check_id = check_id
        self._metrics = metrics or {}

    def model_dump(self) -> dict[str, object]:
        return {"id": self._check_id, "metrics": dict(self._metrics)}


class _FakeMASTClient:
    def __init__(self, *args, **kwargs) -> None:
        self._tic = None

    # LC API (cache-only)
    def search_lightcurve_cached(self, tic_id: int):
        self._tic = tic_id
        return [_FakeSearchRow(sector=1)]

    def download_lightcurve_cached(self, tic_id: int, sector: int, flux_type: str, exptime: float):
        # Provide a minimal LC that spans the ephemeris window.
        time = np.arange(0.0, 30.0, 0.02, dtype=np.float64)
        flux = np.ones_like(time, dtype=np.float64)
        flux_err = np.full_like(time, 2e-4, dtype=np.float64)
        quality = np.zeros_like(time, dtype=np.int32)
        valid = np.ones_like(time, dtype=bool)
        from tess_vetter.domain.lightcurve import LightCurveData

        return LightCurveData(
            time=time,
            flux=flux,
            flux_err=flux_err,
            quality=quality,
            valid_mask=valid,
            tic_id=tic_id,
            sector=sector,
            cadence_seconds=120.0,
        )

    # TPF API (cache-only)
    def search_tpf_cached(self, tic_id: int):
        # Two cached sectors exist: sector 1 has no transit coverage, sector 2 does.
        return [_FakeSearchRow(sector=1), _FakeSearchRow(sector=2)]

    def download_tpf_cached(self, tic_id: int, sector: int, exptime: float):
        n = 50
        stamp = (7, 7)
        quality = np.zeros(n, dtype=np.int32)
        aperture = np.zeros(stamp, dtype=np.int32)
        aperture[2:5, 2:5] = 1

        # Candidate ephemeris for the test: t0=1.0, P=5d, duration=2h.
        # Sector 1 time range avoids transit; sector 2 includes multiple transits.
        time = (
            np.linspace(0.2, 0.8, n) if sector == 1 else np.linspace(0.95, 1.05, n)
        )  # far from phase 0 vs straddles t0

        cube = np.zeros((n, stamp[0], stamp[1]), dtype=np.float64)
        # OOT baseline
        cube[:] = 1000.0
        # In-transit: dim the aperture pixels to create a real difference image
        phase = ((time - 1.0) % 5.0) / 5.0
        phase = np.where(phase > 0.5, phase - 1.0, phase)
        half = ((2.0 / 24.0) / 2.0) / 5.0
        in_tr = np.abs(phase) <= half
        if np.any(in_tr):
            cube[in_tr][:, aperture.astype(bool)] -= 10.0

        flux_err = np.ones_like(cube) * 1.0
        wcs = None
        return time, cube, flux_err, wcs, aperture, quality


def test_pipeline_tries_other_tpf_sector_when_first_has_no_transit_coverage(monkeypatch) -> None:
    # Patch the API MASTClient used by enrich_candidate.
    import tess_vetter.api.io as io_api

    monkeypatch.setattr(io_api, "MASTClient", _FakeMASTClient)

    # Patch vetting to avoid running the full check suite; keep it deterministic.
    import importlib

    vet_mod = importlib.import_module("tess_vetter.api.vet")

    def _fake_vet(*args, **kwargs):
        checks = [_FakeCheck("V09", {"distance_to_target_pixels": 0.2, "localization_reliable": True})]
        return _FakeBundle(results=checks, warnings=[], inputs_summary={}, provenance={"duration_ms": 0.0})

    monkeypatch.setattr(vet_mod, "vet_candidate", _fake_vet)

    cfg = FeatureConfig(
        bulk_mode=True,
        network_ok=False,
        no_download=True,
        cache_dir="/tmp/fake-cache",
    )

    raw, row = enrich_candidate(
        tic_id=999,
        toi=None,
        period_days=5.0,
        t0_btjd=1.0,
        duration_hours=2.0,
        depth_ppm=1000.0,
        config=cfg,
        sectors=[1],
    )

    assert row["status"] == "OK"
    attempts = raw["provenance"]["tpf_attempts"]
    # sector 1 should be rejected for coverage, then sector 2 accepted
    assert any(a.get("sector") == 1 and a.get("reason") == "no_transit_coverage" for a in attempts)
    assert any(a.get("sector") == 2 and a.get("ok") is True for a in attempts)
    assert raw["provenance"]["tpf_sector_used"] == 2

    # Ensure JSON serializability for logging / JSONL output.
    json.dumps(raw)
    json.dumps(row)
