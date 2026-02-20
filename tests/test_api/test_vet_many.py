from __future__ import annotations

import numpy as np

from tess_vetter.api import Candidate, Ephemeris, LightCurve, vet_many


def test_vet_many_returns_bundles_and_summary_rows() -> None:
    time = np.linspace(0, 10, 500)
    flux = np.ones_like(time)
    flux_err = np.ones_like(time) * 1e-3

    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)

    candidates = [
        Candidate(
            ephemeris=Ephemeris(period_days=3.0, t0_btjd=1.0, duration_hours=2.0), depth_ppm=500
        ),
        Candidate(
            ephemeris=Ephemeris(period_days=5.0, t0_btjd=2.0, duration_hours=2.0), depth_ppm=800
        ),
    ]

    bundles, summary = vet_many(lc, candidates, network=False, checks=["V01", "V02"])

    assert len(bundles) == 2
    assert len(summary) == 2

    assert summary[0]["candidate_index"] == 0
    assert summary[0]["period_days"] == 3.0
    assert summary[1]["candidate_index"] == 1
    assert summary[1]["period_days"] == 5.0

    for bundle in bundles:
        assert bundle.results
        assert all(r.id in {"V01", "V02"} for r in bundle.results)
