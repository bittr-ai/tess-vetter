from __future__ import annotations

import numpy as np

from tess_vetter.api import Candidate, Ephemeris, LightCurve
from tess_vetter.api.contracts import model_output_schema
from tess_vetter.api.vet import (
    VET_CANDIDATE_CALL_SCHEMA,
    VET_CANDIDATE_OUTPUT_SCHEMA,
    VET_MANY_CALL_SCHEMA,
    VET_MANY_OUTPUT_SCHEMA,
    VET_MANY_SUMMARY_ROW_SCHEMA,
    vet_candidate,
    vet_many,
)
from tess_vetter.validation.result_schema import VettingBundleResult


def _demo_lightcurve() -> LightCurve:
    time = np.linspace(0.0, 10.0, 500)
    flux = np.ones_like(time)
    flux_err = np.ones_like(time) * 1e-3
    return LightCurve(time=time, flux=flux, flux_err=flux_err)


def _demo_candidate() -> Candidate:
    return Candidate(
        ephemeris=Ephemeris(period_days=3.0, t0_btjd=1.0, duration_hours=2.0),
        depth_ppm=500.0,
    )


def test_vet_call_schema_constants_are_stable() -> None:
    assert VET_CANDIDATE_CALL_SCHEMA == {
        "type": "object",
        "properties": {
            "candidate": {},
            "checks": {},
            "context": {},
            "dec_deg": {},
            "lc": {},
            "network": {},
            "pipeline_config": {},
            "preset": {},
            "ra_deg": {},
            "stellar": {},
            "tic_id": {},
            "tpf": {},
        },
        "additionalProperties": False,
        "required": ["candidate", "lc"],
    }
    assert VET_MANY_CALL_SCHEMA == {
        "type": "object",
        "properties": {
            "candidates": {},
            "checks": {},
            "context": {},
            "dec_deg": {},
            "lc": {},
            "network": {},
            "pipeline_config": {},
            "preset": {},
            "ra_deg": {},
            "stellar": {},
            "tic_id": {},
            "tpf": {},
        },
        "additionalProperties": False,
        "required": ["candidates", "lc"],
    }


def test_vet_schema_constants_match_contract_helpers() -> None:
    assert model_output_schema(VettingBundleResult) == VET_CANDIDATE_OUTPUT_SCHEMA
    assert VET_MANY_SUMMARY_ROW_SCHEMA == {
        "type": "object",
        "properties": {
            "candidate_index": {"type": "integer"},
            "period_days": {"type": "number"},
            "t0_btjd": {"type": "number"},
            "duration_hours": {"type": "number"},
            "depth_ppm": {"type": "number"},
            "n_ok": {"type": "integer"},
            "n_skipped": {"type": "integer"},
            "n_error": {"type": "integer"},
            "flags_top": {"type": "array", "items": {"type": "string"}},
            "runtime_ms": {"type": ["number", "null"]},
        },
        "required": [
            "candidate_index",
            "period_days",
            "t0_btjd",
            "duration_hours",
            "depth_ppm",
            "n_ok",
            "n_skipped",
            "n_error",
            "flags_top",
            "runtime_ms",
        ],
        "additionalProperties": False,
    }
    assert VET_MANY_OUTPUT_SCHEMA == {
        "type": "array",
        "prefixItems": [
            {"type": "array", "items": VET_CANDIDATE_OUTPUT_SCHEMA},
            {"type": "array", "items": VET_MANY_SUMMARY_ROW_SCHEMA},
        ],
        "items": False,
        "minItems": 2,
        "maxItems": 2,
    }


def test_vet_many_single_candidate_parity_with_vet_candidate() -> None:
    lc = _demo_lightcurve()
    candidate = _demo_candidate()

    single = vet_candidate(lc, candidate, network=False, checks=["V01", "V02"])
    bundles, summary = vet_many(lc, [candidate], network=False, checks=["V01", "V02"])

    assert len(bundles) == 1
    assert len(summary) == 1

    bundled = bundles[0]
    assert [(r.id, r.status) for r in single.results] == [
        (r.id, r.status) for r in bundled.results
    ]
    assert [r.id for r in bundled.results] == ["V01", "V02"]

    row = summary[0]
    assert set(row) == {
        "candidate_index",
        "period_days",
        "t0_btjd",
        "duration_hours",
        "depth_ppm",
        "n_ok",
        "n_skipped",
        "n_error",
        "flags_top",
        "runtime_ms",
    }
    assert row["candidate_index"] == 0
    assert row["period_days"] == 3.0
