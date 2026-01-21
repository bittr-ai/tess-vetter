from __future__ import annotations

import numpy as np
import pytest
from astropy.wcs import WCS

import bittr_tess_vetter.api.io as btv_io
from bittr_tess_vetter.domain.lightcurve import LightCurveData
from bittr_tess_vetter.features import FeatureConfig
from bittr_tess_vetter.pipeline import enrich_candidate


class _DummyTarget:
    ra = None
    dec = None
    stellar = None


class _FakeMASTClient:
    def search_lightcurve(self, tic_id: int) -> list[btv_io.SearchResult]:
        return [btv_io.SearchResult(tic_id=tic_id, sector=1, author="SPOC", exptime=120.0)]

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
        time = np.linspace(90.0, 120.0, 2000, dtype=np.float64)
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
            cadence_seconds=120.0,
        )

    def download_tpf(self, tic_id: int, sector: int, exptime: float = 120.0):
        # Build a tiny stamp where one pixel contains a transit-like dip.
        n = 1000
        time = np.linspace(99.5, 100.5, n, dtype=np.float64)
        flux = np.ones((n, 5, 5), dtype=np.float64)
        flux_err = np.ones_like(flux) * 1e-3
        # Dip the center pixel in transit
        duration_days = 2.0 / 24.0
        in_transit = np.abs(time - 100.0) < (duration_days / 2.0)
        flux[in_transit, 2, 2] -= 0.01

        wcs = WCS(naxis=2)
        aperture_mask = np.zeros((5, 5), dtype=np.int32)
        aperture_mask[2, 2] = 1
        quality = np.zeros(n, dtype=np.int32)
        return time, flux, flux_err, wcs, aperture_mask, quality

    def get_target_info(self, tic_id: int) -> _DummyTarget:
        return _DummyTarget()


def test_pipeline_populates_localization_and_ghost(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(btv_io, "MASTClient", lambda: _FakeMASTClient())  # type: ignore[assignment]

    _raw, row = enrich_candidate(
        123,
        toi=None,
        period_days=10.0,
        t0_btjd=100.0,
        duration_hours=2.0,
        depth_ppm=500.0,
        config=FeatureConfig(network_ok=True, bulk_mode=True, allow_20s=False, require_tpf=True),
        sectors=[1],
    )

    assert row["status"] == "OK"
    assert row["localization_verdict"] in {
        "ON_TARGET",
        "OFF_TARGET",
        "AMBIGUOUS",
        "INVALID",
        "on_target",
        "off_target",
        "ambiguous",
        "invalid",
    }
    assert row["ghost_like_score_adjusted_median"] is not None
    assert row["scattered_light_risk_median"] is not None
