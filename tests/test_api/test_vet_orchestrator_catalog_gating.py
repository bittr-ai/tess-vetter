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
    """Test that V06 (nearby EB search) is skipped when coordinates are missing.

    Uses new API: `checks=["V06"]` instead of old `enabled={"V06"}`.
    The new CheckResult has `flags` instead of `details`, with format "SKIPPED:{reason}".
    """
    result = vet_candidate(
        _minimal_lc(),
        _minimal_candidate(),
        checks=["V06"],
        network=False,
        ra_deg=None,
        dec_deg=None,
        tic_id=None,
    )

    # Find the V06 result
    v06_results = [r for r in result.results if r.id == "V06"]
    assert len(v06_results) == 1
    v06 = v06_results[0]
    assert v06.status == "skipped"
    # New schema: reason is in flags list with format "SKIPPED:{reason}"
    skip_flag = [f for f in v06.flags if f.startswith("SKIPPED:")]
    assert len(skip_flag) == 1
    reason = skip_flag[0].replace("SKIPPED:", "")
    assert reason in ["NO_COORDINATES", "NO_RA_DEC", "NETWORK_DISABLED"]
    assert any("V06" in w for w in result.warnings)


def test_vet_candidate_v07_missing_tic_id_returns_skipped_result() -> None:
    """Test that V07 (ExoFOP lookup) is skipped when TIC ID is missing.

    Uses new API: `checks=["V07"]` instead of old `enabled={"V07"}`.
    The new CheckResult has `flags` instead of `details`, with format "SKIPPED:{reason}".
    """
    result = vet_candidate(
        _minimal_lc(),
        _minimal_candidate(),
        checks=["V07"],
        network=False,
        ra_deg=None,
        dec_deg=None,
        tic_id=None,
    )

    # Find the V07 result
    v07_results = [r for r in result.results if r.id == "V07"]
    assert len(v07_results) == 1
    v07 = v07_results[0]
    assert v07.status == "skipped"
    # New schema: reason is in flags list with format "SKIPPED:{reason}"
    skip_flag = [f for f in v07.flags if f.startswith("SKIPPED:")]
    assert len(skip_flag) == 1
    reason = skip_flag[0].replace("SKIPPED:", "")
    assert reason in ["NO_TIC_ID", "NETWORK_DISABLED"]
    assert any("V07" in w for w in result.warnings)

