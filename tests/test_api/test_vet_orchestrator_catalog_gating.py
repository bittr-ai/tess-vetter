from __future__ import annotations

import numpy as np

from bittr_tess_vetter.api.types import Candidate, Ephemeris, LightCurve
from bittr_tess_vetter.api.vet import vet_candidate


def _minimal_lc() -> LightCurve:
    time = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    flux = np.ones_like(time)
    flux_err = np.full_like(time, 1e-3)
    return LightCurve(time=time, flux=flux, flux_err=flux_err)


def _minimal_candidate() -> Candidate:
    eph = Ephemeris(period_days=2.0, t0_btjd=0.0, duration_hours=2.0)
    return Candidate(ephemeris=eph, depth_ppm=1000.0)


def test_vet_candidate_v06_missing_ra_dec_returns_skipped_result() -> None:
    result = vet_candidate(
        _minimal_lc(),
        _minimal_candidate(),
        enabled={"V06"},
        network=False,
        ra_deg=None,
        dec_deg=None,
        tic_id=None,
    )

    v06 = result.get_result("V06")
    assert v06 is not None
    assert v06.details["status"] == "skipped"
    assert v06.details["reason"] == "missing_metadata"
    assert "ra_deg" in v06.details["missing"]
    assert "dec_deg" in v06.details["missing"]
    assert any("V06" in w for w in result.warnings)


def test_vet_candidate_v07_missing_tic_id_returns_skipped_result() -> None:
    result = vet_candidate(
        _minimal_lc(),
        _minimal_candidate(),
        enabled={"V07"},
        network=False,
        ra_deg=None,
        dec_deg=None,
        tic_id=None,
    )

    v07 = result.get_result("V07")
    assert v07 is not None
    assert v07.details["status"] == "skipped"
    assert v07.details["reason"] == "missing_metadata"
    assert v07.details["missing"] == ["tic_id"]
    assert any("V07" in w for w in result.warnings)

