from __future__ import annotations

import numpy as np
import pytest
from astropy.wcs import WCS

import tess_vetter.api.io as btv_io
from tess_vetter.domain.lightcurve import LightCurveData
from tess_vetter.features import FeatureConfig
from tess_vetter.pipeline import enrich_candidate


class _DummyTarget:
    ra = None
    dec = None
    stellar = None


class _FakeMASTClientNoWindows:
    """TPF time range does not include any predicted transits => NO_EVIDENCE windows."""

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
        # Time window far away from t0=100.0 below (no transits land in this TPF).
        n = 500
        time = np.linspace(0.0, 1.0, n, dtype=np.float64)
        flux = np.ones((n, 5, 5), dtype=np.float64)
        flux_err = np.ones_like(flux) * 1e-3
        wcs = WCS(naxis=2)
        aperture_mask = np.zeros((5, 5), dtype=np.int32)
        aperture_mask[2, 2] = 1
        quality = np.zeros(n, dtype=np.int32)
        return time, flux, flux_err, wcs, aperture_mask, quality

    def get_target_info(self, tic_id: int) -> _DummyTarget:
        return _DummyTarget()


def test_pipeline_pixel_timeseries_no_evidence_sets_delta_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(btv_io, "MASTClient", lambda: _FakeMASTClientNoWindows())  # type: ignore[assignment]

    _raw, row = enrich_candidate(
        123,
        toi=None,
        # Choose a period long enough that no predicted transits fall within the TPF time window.
        period_days=1000.0,
        t0_btjd=100.0,
        duration_hours=2.0,
        depth_ppm=500.0,
        # network_ok=True so the pipeline is allowed to call the (patched) MAST client.
        config=FeatureConfig(network_ok=True, bulk_mode=True, allow_20s=False, require_tpf=True),
        sectors=[1],
    )

    assert row["status"] == "OK"
    assert row["pixel_timeseries_verdict"] == "NO_EVIDENCE"
    assert row["pixel_timeseries_delta_chi2"] == 0.0
