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


def test_vet_candidate_returns_vetting_bundle_result() -> None:
    """Test that vet_candidate returns a VettingBundleResult.

    Note: The old API with `enabled` and `policy_mode` parameters has been removed
    in v0.1.0. Use VettingPipeline directly for advanced configuration.
    """
    time = np.linspace(0.0, 10.0, 500, dtype=np.float64)
    flux = np.ones_like(time)
    flux_err = np.full_like(time, 1e-3)
    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    cand = Candidate(ephemeris=Ephemeris(period_days=2.0, t0_btjd=0.5, duration_hours=2.0), depth_ppm=1000.0)

    # The new API always returns metrics-only results
    out = vet_candidate(lc, cand, checks=["V01"])
    assert hasattr(out, "results")
    assert hasattr(out, "provenance")
