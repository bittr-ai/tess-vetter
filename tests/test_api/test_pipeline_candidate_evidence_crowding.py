from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

import bittr_tess_vetter.api.io as btv_io
from bittr_tess_vetter.domain.lightcurve import LightCurveData
from bittr_tess_vetter.features import FeatureConfig
from bittr_tess_vetter.pipeline import enrich_candidate
from bittr_tess_vetter.platform.catalogs.gaia_client import (
    GaiaAstrophysicalParams,
    GaiaNeighbor,
    GaiaQueryResult,
    GaiaSourceRecord,
)
from bittr_tess_vetter.platform.catalogs.models import SourceRecord


class _DummyTarget:
    ra = 10.0
    dec = 20.0
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

    def get_target_info(self, tic_id: int) -> _DummyTarget:
        return _DummyTarget()


def test_candidate_evidence_gaia_crowding(monkeypatch: pytest.MonkeyPatch) -> None:
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
            phot_g_mean_mag=12.0,
            delta_mag=2.0,
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

    import bittr_tess_vetter.platform.catalogs.gaia_client as gaia_mod

    monkeypatch.setattr(gaia_mod, "query_gaia_by_position_sync", _fake_gaia)  # type: ignore[assignment]

    _raw, row = enrich_candidate(
        123,
        toi=None,
        period_days=10.0,
        t0_btjd=100.0,
        duration_hours=2.0,
        depth_ppm=None,
        config=FeatureConfig(network_ok=True, bulk_mode=True, require_tpf=False),
        sectors=[1],
    )

    assert row["status"] == "OK"
    assert row["n_gaia_neighbors_21arcsec"] == 1
    assert row["brightest_neighbor_delta_mag"] == 2.0
    # target flux fraction = 1/(1+10^(-0.4*2)) ~ 0.863; crowding_metric = 1 - that ~ 0.137
    assert row["crowding_metric"] is not None
    assert abs(float(row["crowding_metric"]) - 0.137) < 0.02

