from __future__ import annotations

import numpy as np
import pytest

import bittr_tess_vetter.api.io as btv_io
from bittr_tess_vetter.domain.lightcurve import LightCurveData
from bittr_tess_vetter.features import FeatureConfig
from bittr_tess_vetter.pipeline import enrich_candidate


class _DummyTarget:
    ra = None
    dec = None
    stellar = None


class _FakeMASTClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, float | None]] = []

    def search_lightcurve(self, tic_id: int) -> list[btv_io.SearchResult]:
        # sector 1 has 120s, sector 2 only has 20s
        return [
            btv_io.SearchResult(tic_id=tic_id, sector=1, author="SPOC", exptime=120.0),
            btv_io.SearchResult(tic_id=tic_id, sector=2, author="SPOC", exptime=20.0),
        ]

    def download_lightcurve(
        self,
        tic_id: int,
        sector: int,
        flux_type: str = "pdcsap",
        quality_mask: int | None = None,
        exptime: float | None = None,
        author: str | None = None,
        progress_callback=None,
    ) -> LightCurveData:
        self.calls.append(("lc", int(sector), exptime))
        time = np.linspace(90.0, 120.0, 500, dtype=np.float64)
        flux = np.ones_like(time)
        flux_err = np.ones_like(time) * 5e-4
        quality = np.zeros_like(time, dtype=np.int32)
        valid_mask = np.ones_like(time, dtype=bool)
        return LightCurveData(
            time=time,
            flux=flux,
            flux_err=flux_err,
            quality=quality,
            valid_mask=valid_mask,
            tic_id=int(tic_id),
            sector=int(sector),
            cadence_seconds=float(exptime or 120.0),
        )

    def get_target_info(self, tic_id: int) -> _DummyTarget:
        return _DummyTarget()

    def download_tpf(self, tic_id: int, sector: int, exptime: float = 120.0):
        raise btv_io.LightCurveNotFoundError("no tpf")


def test_per_sector_exptime_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeMASTClient()
    monkeypatch.setattr(btv_io, "MASTClient", lambda: fake)  # type: ignore[assignment]

    _raw, row = enrich_candidate(
        123,
        toi=None,
        period_days=10.0,
        t0_btjd=100.0,
        duration_hours=2.0,
        depth_ppm=500.0,
        config=FeatureConfig(network_ok=True, allow_20s=True, bulk_mode=True, require_tpf=False),
    )

    assert row["status"] == "OK"
    # sector 1 should request 120s, sector 2 should request 20s (allow_20s=True)
    assert ("lc", 1, 120.0) in fake.calls
    assert ("lc", 2, 20.0) in fake.calls


def test_require_tpf_fails_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeMASTClient()
    monkeypatch.setattr(btv_io, "MASTClient", lambda: fake)  # type: ignore[assignment]

    _raw, row = enrich_candidate(
        123,
        toi=None,
        period_days=10.0,
        t0_btjd=100.0,
        duration_hours=2.0,
        depth_ppm=500.0,
        config=FeatureConfig(network_ok=True, allow_20s=True, bulk_mode=True, require_tpf=True),
    )

    assert row["status"] == "ERROR"
    assert row["error_class"] == "TPFRequiredError"

