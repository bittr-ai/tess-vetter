from __future__ import annotations

from dataclasses import dataclass

import pytest

import tess_vetter.api.catalog as catalog_api
from tess_vetter.api.catalog import vet_catalog
from tess_vetter.api.types import Candidate, CheckResult, Ephemeris


@dataclass
class _FakeInternalCatalogResult:
    id: str
    name: str
    confidence: float
    details: dict[str, object]


@pytest.fixture
def candidate() -> Candidate:
    return Candidate(
        ephemeris=Ephemeris(period_days=3.0, t0_btjd=1001.0, duration_hours=2.0), depth_ppm=1000
    )


def test_vet_catalog_network_disabled_returns_skipped(candidate: Candidate) -> None:
    results = vet_catalog(candidate, tic_id=123, ra_deg=10.0, dec_deg=-20.0, network=False)
    assert len(results) == 2
    assert all(isinstance(r, CheckResult) for r in results)
    assert results[0].details.get("status") == "skipped"
    assert results[1].details.get("status") == "skipped"


def test_vet_catalog_missing_metadata_returns_missing_metadata(candidate: Candidate) -> None:
    results = vet_catalog(candidate, network=True)  # missing tic_id, ra_deg, dec_deg
    assert len(results) == 2
    assert results[0].id == "V06"
    assert results[0].details.get("reason") == "missing_metadata"
    assert "ra_deg" in (results[0].details.get("missing") or [])
    assert "dec_deg" in (results[0].details.get("missing") or [])
    assert results[1].id == "V07"
    assert results[1].details.get("reason") == "missing_metadata"
    assert results[1].details.get("missing") == ["tic_id"]


def test_nearby_eb_search_converts_non_scalar_details_to_raw(
    monkeypatch: pytest.MonkeyPatch, candidate: Candidate
) -> None:
    def _fake_run_nearby_eb_search(**_: object) -> _FakeInternalCatalogResult:
        return _FakeInternalCatalogResult(
            id="V06",
            name="nearby_eb_search",
            confidence=0.81,
            details={
                "status": "ok",
                "n_ebs_found": 2,
                "complex_payload": {"matches": [1, 2, 3]},
                "list_payload": [1, 2, 3],
            },
        )

    monkeypatch.setattr(catalog_api, "run_nearby_eb_search", _fake_run_nearby_eb_search)

    result = catalog_api.nearby_eb_search(
        candidate,
        ra_deg=10.0,
        dec_deg=-20.0,
        network=True,
    )

    assert result.status == "ok"
    assert result.metrics["status"] == "ok"
    assert result.metrics["n_ebs_found"] == 2
    assert "complex_payload" not in result.metrics
    assert "list_payload" not in result.metrics
    assert result.raw is not None
    assert result.raw["complex_payload"] == {"matches": [1, 2, 3]}
    assert result.raw["list_payload"] == [1, 2, 3]
