from __future__ import annotations

from bittr_tess_vetter.api.catalog import vet_catalog
from bittr_tess_vetter.api.types import Candidate, CheckResult, Ephemeris


def test_vet_catalog_network_disabled_returns_skipped() -> None:
    cand = Candidate(ephemeris=Ephemeris(period_days=3.0, t0_btjd=1001.0, duration_hours=2.0), depth_ppm=1000)
    results = vet_catalog(cand, tic_id=123, ra_deg=10.0, dec_deg=-20.0, network=False)
    assert len(results) == 2
    assert all(isinstance(r, CheckResult) for r in results)
    assert results[0].details.get("status") == "skipped"
    assert results[1].details.get("status") == "skipped"


def test_vet_catalog_missing_metadata_returns_missing_metadata() -> None:
    cand = Candidate(ephemeris=Ephemeris(period_days=3.0, t0_btjd=1001.0, duration_hours=2.0), depth_ppm=1000)
    results = vet_catalog(cand, network=True)  # missing tic_id, ra_deg, dec_deg
    assert len(results) == 2
    assert results[0].id == "V06"
    assert results[0].details.get("reason") == "missing_metadata"
    assert "ra_deg" in (results[0].details.get("missing") or [])
    assert "dec_deg" in (results[0].details.get("missing") or [])
    assert results[1].id == "V07"
    assert results[1].details.get("reason") == "missing_metadata"
    assert results[1].details.get("missing") == ["tic_id"]

