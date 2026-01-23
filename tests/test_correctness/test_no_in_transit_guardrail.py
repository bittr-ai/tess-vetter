from __future__ import annotations

import numpy as np

import bittr_tess_vetter.api.io as btv_io
from bittr_tess_vetter.domain.lightcurve import LightCurveData
from bittr_tess_vetter.features import FeatureConfig
from bittr_tess_vetter.pipeline import enrich_candidate


class _FakeSearchRow:
    def __init__(self, sector: int, exptime: float = 120.0) -> None:
        self.sector = sector
        self.exptime = exptime


class _FakeMASTClient:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def search_lightcurve_cached(self, tic_id: int):
        return [_FakeSearchRow(sector=1)]

    def download_lightcurve_cached(self, tic_id: int, sector: int, flux_type: str, exptime: float):
        # Construct a time series where t0 lies within the baseline, but the cadence
        # is too coarse relative to an extremely short duration, so no in-transit
        # cadences occur (n_in_transit=0).
        #
        # Ephemeris for the test: t0=5.0, P=10d, duration=0.01h (36s).
        # Half-duration window is ~18s; we offset samples by ~26s from t0.
        t0 = 5.0
        time = t0 - 0.0003 + np.arange(0, 500, dtype=np.float64) * 0.001
        flux = np.ones_like(time, dtype=np.float64)
        flux_err = np.ones_like(time, dtype=np.float64) * 1e-4
        quality = np.zeros_like(time, dtype=np.int32)
        valid = np.ones_like(time, dtype=bool)
        return LightCurveData(
            time=time,
            flux=flux,
            flux_err=flux_err,
            quality=quality,
            valid_mask=valid,
            tic_id=int(tic_id),
            sector=int(sector),
            cadence_seconds=120.0,
        )


def test_no_in_transit_cadences_returns_error(monkeypatch) -> None:
    monkeypatch.setattr(btv_io, "MASTClient", _FakeMASTClient)

    _raw, row = enrich_candidate(
        123,
        toi=None,
        period_days=10.0,
        t0_btjd=5.0,
        duration_hours=0.01,
        depth_ppm=500.0,
        config=FeatureConfig(network_ok=False, no_download=True, cache_dir="/tmp/fake-cache"),
        sectors=[1],
    )

    assert row["status"] == "ERROR"
    assert row["error_class"] == "NoInTransitCadencesError"
