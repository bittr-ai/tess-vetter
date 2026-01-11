from __future__ import annotations

import numpy as np
import pytest

from bittr_tess_vetter.api.lc_only import odd_even_depth
from bittr_tess_vetter.api.types import Candidate, Ephemeris, LightCurve
from bittr_tess_vetter.api.vet import vet_candidate


def test_lc_only_policy_mode_is_deprecated_and_ignored() -> None:
    time = np.linspace(0.0, 10.0, 500, dtype=np.float64)
    flux = np.ones_like(time)
    flux_err = np.full_like(time, 1e-3)
    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    eph = Ephemeris(period_days=3.5, t0_btjd=0.5, duration_hours=2.5)

    with pytest.warns(FutureWarning, match="policy_mode"):
        r = odd_even_depth(lc, eph, policy_mode="anything")
    assert r.passed is None
    assert r.details.get("_metrics_only") is True


def test_vet_candidate_policy_mode_is_deprecated_and_provenance_records_request() -> None:
    time = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    flux = np.ones_like(time)
    flux_err = np.full_like(time, 1e-3)
    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    cand = Candidate(ephemeris=Ephemeris(period_days=2.0, t0_btjd=0.0, duration_hours=2.0), depth_ppm=1000.0)

    with pytest.warns(FutureWarning, match="policy_mode"):
        out = vet_candidate(lc, cand, enabled={"V01"}, policy_mode="not_supported")
    assert out.provenance["policy_mode"] == "metrics_only"
    assert out.provenance["policy_mode_requested"] == "not_supported"
