from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest
from astropy.wcs import WCS

import tess_vetter.api.io as btv_io
from tess_vetter.domain.lightcurve import LightCurveData
from tess_vetter.features import FeatureConfig
from tess_vetter.pipeline import enrich_candidate
from tess_vetter.platform.catalogs.gaia_client import (
    GaiaAstrophysicalParams,
    GaiaNeighbor,
    GaiaQueryResult,
    GaiaSourceRecord,
)
from tess_vetter.platform.catalogs.models import SourceRecord


class _DummyStellar:
    radius = 1.0


class _DummyTarget:
    ra = 10.0
    dec = 20.0
    stellar = _DummyStellar()


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
        n = 1000
        time = np.linspace(99.5, 100.5, n, dtype=np.float64)
        flux = np.ones((n, 5, 5), dtype=np.float64)
        flux_err = np.ones_like(flux) * 1e-3
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


def test_host_plausibility_populates_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(btv_io, "MASTClient", lambda: _FakeMASTClient())  # type: ignore[assignment]

    def _fake_gaia(*_args: object, **_kwargs: object) -> GaiaQueryResult:
        now = datetime.now(UTC)
        source = GaiaSourceRecord(
            source_id=111,
            ra=10.0,
            dec=20.0,
            parallax=1.0,
            parallax_error=0.1,
            pmra=0.0,
            pmdec=0.0,
            phot_g_mean_mag=10.0,
            phot_bp_mean_mag=None,
            phot_rp_mean_mag=None,
            bp_rp=None,
            ruwe=1.0,
            duplicated_source=False,
            non_single_star=None,
            astrometric_excess_noise=None,
            phot_bp_rp_excess_factor=None,
        )
        astrophysical = GaiaAstrophysicalParams(source_id=111, radius_gspphot=1.0)
        neighbor = GaiaNeighbor(
            source_id=222,
            ra=10.0,
            dec=20.0,
            separation_arcsec=10.0,
            phot_g_mean_mag=10.0,
            delta_mag=0.0,
            ruwe=1.0,
        )
        return GaiaQueryResult(
            source=source,
            astrophysical=astrophysical,
            neighbors=[neighbor],
            source_record=SourceRecord(
                name="gaia_dr3",
                version="dr3",
                retrieved_at=now,
                query="test",
            ),
        )

    import tess_vetter.platform.catalogs.gaia_client as gaia_mod

    monkeypatch.setattr(gaia_mod, "query_gaia_by_position_sync", _fake_gaia)  # type: ignore[assignment]

    _raw, row = enrich_candidate(
        123,
        toi=None,
        period_days=10.0,
        t0_btjd=100.0,
        duration_hours=2.0,
        depth_ppm=500.0,
        config=FeatureConfig(
            network_ok=True,
            bulk_mode=True,
            require_tpf=True,
            enable_host_plausibility=True,
        ),
        sectors=[1],
    )

    assert row["status"] == "OK"
    # Host ambiguity within 1 pixel should trigger resolved followup
    assert row["host_requires_resolved_followup"] is True
    assert row["host_physically_impossible_count"] == 0

