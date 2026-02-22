from __future__ import annotations

import numpy as np

from tess_vetter.api.types import Candidate, Ephemeris, LightCurve
from tess_vetter.api.workflow import (
    WORKFLOW_PROVENANCE_SCHEMA_VERSION,
    WORKFLOW_SCHEMA_VERSION,
    WorkflowResultPayload,
    run_candidate_workflow,
)


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


def test_workflow_contract_constants_are_stable() -> None:
    assert WORKFLOW_SCHEMA_VERSION == 1
    assert WORKFLOW_PROVENANCE_SCHEMA_VERSION == 1


def test_run_candidate_workflow_payload_contract_is_compatible() -> None:
    lc_by_sector = _two_sector_lcs()
    candidate = Candidate(
        ephemeris=Ephemeris(period_days=2.0, t0_btjd=0.5, duration_hours=2.0),
        depth_ppm=1000.0,
    )

    result = run_candidate_workflow(
        lc_by_sector=lc_by_sector,
        candidate=candidate,
        checks=["V01"],
        run_per_sector=True,
    )
    payload: WorkflowResultPayload = result.to_dict()

    assert result.schema_version == WORKFLOW_SCHEMA_VERSION
    assert payload["schema_version"] == WORKFLOW_SCHEMA_VERSION
    assert payload["provenance"]["schema_version"] == WORKFLOW_PROVENANCE_SCHEMA_VERSION
    assert payload["stitched"] is not None
    assert payload["stitched"]["sectors"] == [1, 2]
    assert payload["per_sector"] is not None
