from __future__ import annotations

import numpy as np

from tess_vetter.api.per_sector import (
    PER_SECTOR_PROVENANCE_SCHEMA_VERSION,
    PER_SECTOR_SCHEMA_VERSION,
    per_sector_vet,
)
from tess_vetter.api.types import Candidate, Ephemeris, LightCurve


def _two_sector_lcs() -> dict[int, LightCurve]:
    time1 = np.linspace(0.0, 10.0, 400, dtype=np.float64)
    time2 = np.linspace(20.0, 30.0, 400, dtype=np.float64)
    flux1 = np.ones_like(time1)
    flux2 = np.ones_like(time2)
    err1 = np.full_like(time1, 1e-3)
    err2 = np.full_like(time2, 1e-3)
    return {
        1: LightCurve(time=time1, flux=flux1, flux_err=err1),
        2: LightCurve(time=time2, flux=flux2, flux_err=err2),
    }


def _candidate() -> Candidate:
    return Candidate(ephemeris=Ephemeris(period_days=2.0, t0_btjd=0.5, duration_hours=2.0), depth_ppm=1000.0)


def test_per_sector_vet_payload_contract_types() -> None:
    out = per_sector_vet(_two_sector_lcs(), _candidate(), checks=["V01"])

    payload = out.to_dict()
    assert payload["schema_version"] == PER_SECTOR_SCHEMA_VERSION
    assert sorted(payload["bundles_by_sector"].keys()) == [1, 2]

    summary = payload["summary_records"]
    assert len(summary) == 2
    assert summary[0]["sector"] == 1
    assert all(set(row.keys()) == {"sector", "checks", "ok", "error", "skipped"} for row in summary)
    assert all(isinstance(row["checks"], int) for row in summary)

    provenance = payload["provenance"]
    assert provenance["schema_version"] == PER_SECTOR_PROVENANCE_SCHEMA_VERSION
    assert provenance["checks"] == ["V01"]
    assert provenance["sectors"] == [1, 2]
    assert isinstance(provenance["has_tpf_by_sector"], bool)


def test_per_sector_vet_payload_isolation() -> None:
    out = per_sector_vet(_two_sector_lcs(), _candidate(), checks=["V01"])

    payload = out.to_dict()
    payload["summary_records"][0]["sector"] = 999
    payload["provenance"]["sectors"].append(999)

    fresh = out.to_dict()
    assert fresh["summary_records"][0]["sector"] == 1
    assert fresh["provenance"]["sectors"] == [1, 2]
