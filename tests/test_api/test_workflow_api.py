from __future__ import annotations

import numpy as np

from bittr_tess_vetter.api.per_sector import per_sector_vet
from bittr_tess_vetter.api.types import Candidate, Ephemeris, LightCurve
from bittr_tess_vetter.api.workflow import run_candidate_workflow


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


def test_per_sector_vet_smoke() -> None:
    lc_by_sector = _two_sector_lcs()
    cand = Candidate(ephemeris=Ephemeris(period_days=2.0, t0_btjd=0.5, duration_hours=2.0), depth_ppm=1000.0)

    out = per_sector_vet(lc_by_sector, cand, checks=["V01"])
    assert sorted(out.bundles_by_sector.keys()) == [1, 2]
    assert len(out.sector_ephemeris_metrics) == 2
    assert out.summary_records[0]["sector"] == 1


def test_run_candidate_workflow_stitches_and_runs_per_sector() -> None:
    lc_by_sector = _two_sector_lcs()
    cand = Candidate(ephemeris=Ephemeris(period_days=2.0, t0_btjd=0.5, duration_hours=2.0), depth_ppm=1000.0)

    out = run_candidate_workflow(lc_by_sector=lc_by_sector, candidate=cand, checks=["V01"], run_per_sector=True)
    assert out.bundle.get_result("V01") is not None
    assert out.per_sector is not None
    assert sorted(out.per_sector.bundles_by_sector.keys()) == [1, 2]
    assert out.stitched is not None

