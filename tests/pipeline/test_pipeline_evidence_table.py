from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tess_vetter.pipeline_composition.executor import _write_evidence_table


def _write_step(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_evidence_table_extracts_mixed_top_level_and_nested_fields(tmp_path: Path) -> None:
    toi = "TOI-MIXED.01"
    toi_dir = tmp_path / toi / "steps"
    model_path = _write_step(toi_dir / "01_model.json", {"verdict": "MODEL_OK"})
    systematics_path = _write_step(toi_dir / "02_systematics.json", {"result": {"verdict": "SYSTEMATICS_OK"}})
    ephemeris_path = _write_step(toi_dir / "03_ephemeris.json", {"verdict": "EPHEMERIS_OK"})
    timing_path = _write_step(toi_dir / "04_timing.json", {"result": {"verdict": "TIMING_OK"}})
    report_path = _write_step(
        toi_dir / "04b_report.json",
        {
            "summary": {
                "stellar_contamination_summary": {"risk_scalar": 0.37},
                "stellar_contamination_risk_scalar": 0.37,
            }
        },
    )
    localize_path = _write_step(
        toi_dir / "05_localize.json",
        {
            "result": {
                "consensus": {"action_hint": "collect_more_imaging"},
                "reliability_summary": {
                    "status": "REVIEW_REQUIRED",
                    "action_hint": "DEFER_HOST_ASSIGNMENT",
                },
            }
        },
    )
    dilution_path = _write_step(
        toi_dir / "06_dilution.json",
        {
            "result": {
                "n_plausible_scenarios": 4,
                "reliability_summary": {
                    "status": "REVIEW_REQUIRED",
                    "action_hint": "REVIEW_WITH_DILUTION",
                },
            }
        },
    )
    fpp_path = _write_step(toi_dir / "07_fpp.json", {"result": {"fpp": 0.017}})
    vet_path = _write_step(
        toi_dir / "09_vet.json",
        {
            "summary": {"known_planet_match_status": "confirmed_same_planet"},
            "known_planet_match": {
                "status": "confirmed_same_planet",
                "matched_planet": {"name": "TOI-411 c", "period": 9.57307},
            },
        },
    )
    neighbors_path = _write_step(
        toi_dir / "08_neighbors.json",
        {"multiplicity_risk": {"status": "ELEVATED", "reasons": ["TARGET_RUWE_ELEVATED"]}},
    )

    toi_result = {
        "toi": toi,
        "concern_flags": [],
        "steps": [
            {"op": "model_compete", "status": "ok", "step_output_path": model_path},
            {"op": "systematics_proxy", "status": "ok", "step_output_path": systematics_path},
            {"op": "ephemeris_reliability", "status": "ok", "step_output_path": ephemeris_path},
            {"op": "timing", "status": "ok", "step_output_path": timing_path},
            {"op": "report", "status": "ok", "step_output_path": report_path},
            {"op": "localize_host", "status": "ok", "step_output_path": localize_path},
            {"op": "dilution", "status": "ok", "step_output_path": dilution_path},
            {"op": "fpp_run", "status": "ok", "step_output_path": fpp_path},
            {"op": "resolve_neighbors", "status": "ok", "step_output_path": neighbors_path},
            {"op": "vet", "status": "ok", "step_output_path": vet_path},
        ],
    }

    rows = _write_evidence_table(out_dir=tmp_path, toi_results=[toi_result])
    row = rows[0]
    assert row["model_compete_verdict"] == "MODEL_OK"
    assert row["systematics_verdict"] == "SYSTEMATICS_OK"
    assert row["ephemeris_verdict"] == "EPHEMERIS_OK"
    assert row["timing_verdict"] == "TIMING_OK"
    assert row["localize_host_action_hint"] == "collect_more_imaging"
    assert row["localize_host_reliability_status"] == "REVIEW_REQUIRED"
    assert row["localize_host_reliability_action_hint"] == "DEFER_HOST_ASSIGNMENT"
    assert row["dilution_n_plausible_scenarios"] == 4
    assert row["dilution_reliability_status"] == "REVIEW_REQUIRED"
    assert row["dilution_reliability_action_hint"] == "REVIEW_WITH_DILUTION"
    assert row["multiplicity_risk_status"] == "ELEVATED"
    assert row["multiplicity_risk_reasons"] == ["TARGET_RUWE_ELEVATED"]
    assert row["fpp"] == 0.017
    assert row["known_planet_status"] == "confirmed_same_planet"
    assert row["known_planet_name"] == "TOI-411 c"
    assert row["known_planet_period"] == pytest.approx(9.57307)
    assert row["stellar_contamination_risk_scalar"] == pytest.approx(0.37)


def test_evidence_table_extracts_contrast_curve_fields_and_selected_metadata(tmp_path: Path) -> None:
    toi = "TOI-CONTRAST.01"
    toi_dir = tmp_path / toi / "steps"
    contrast_curves_path = _write_step(
        toi_dir / "01_contrast_curves.json",
        {
            "result": {
                "availability": "available",
                "n_observations": 3,
                "filter": "Kcont",
                "quality": "high",
                "depth0p5": 4.2,
                "depth1p0": 6.8,
            }
        },
    )
    contrast_summary_path = _write_step(
        toi_dir / "02_contrast_curve_summary.json",
        {
            "summary": {
                "availability": "available",
                "n_observations": 3,
                "filter": "Ks",
                "quality": "selected",
                "depth0p5": 4.5,
                "depth1p0": 7.1,
                "selected_curve": {
                    "id": "curve-1",
                    "source": "exofop",
                    "filter": "Ks",
                    "quality": "A",
                    "depth0p5": 4.5,
                    "depth1p0": 7.1,
                    "facility": "PHARO/P200",
                },
            }
        },
    )

    toi_result = {
        "toi": toi,
        "concern_flags": [],
        "steps": [
            {"op": "contrast_curves", "status": "ok", "step_output_path": contrast_curves_path},
            {"op": "contrast_curve_summary", "status": "ok", "step_output_path": contrast_summary_path},
        ],
    }

    rows = _write_evidence_table(out_dir=tmp_path, toi_results=[toi_result])
    row = rows[0]
    assert row["contrast_curve_availability"] == "available"
    assert row["contrast_curve_n_observations"] == 3
    assert row["contrast_curve_filter"] == "Ks"
    assert row["contrast_curve_quality"] == "selected"
    assert row["contrast_curve_depth0p5"] == pytest.approx(4.5)
    assert row["contrast_curve_depth1p0"] == pytest.approx(7.1)
    assert row["contrast_curve_selected_id"] == "curve-1"
    assert row["contrast_curve_selected_source"] == "exofop"
    assert row["contrast_curve_selected_filter"] == "Ks"
    assert row["contrast_curve_selected_quality"] == "A"
    assert row["contrast_curve_selected_depth0p5"] == pytest.approx(4.5)
    assert row["contrast_curve_selected_depth1p0"] == pytest.approx(7.1)
    assert row["contrast_curve_selected_metadata"] == {
        "id": "curve-1",
        "source": "exofop",
        "filter": "Ks",
        "quality": "A",
        "depth0p5": 4.5,
        "depth1p0": 7.1,
        "facility": "PHARO/P200",
    }


def test_evidence_table_concern_flags_are_deduped_and_csv_sorted(tmp_path: Path) -> None:
    toi = "TOI-FLAGS.01"
    toi_dir = tmp_path / toi / "steps"
    report_path = _write_step(
        toi_dir / "01_report.json",
        {
            "summary": {"concerns": ["zeta", "alpha"]},
            "warnings": ["beta"],
        },
    )
    model_path = _write_step(
        toi_dir / "02_model.json",
        {"verdict": "MODEL_OK", "result": {"warnings": ["gamma"]}},
    )

    toi_result = {
        "toi": toi,
        "concern_flags": ["delta", "alpha"],
        "steps": [
            {"op": "report", "status": "ok", "step_output_path": report_path},
            {"op": "model_compete", "status": "ok", "step_output_path": model_path},
        ],
    }

    rows = _write_evidence_table(out_dir=tmp_path, toi_results=[toi_result])
    assert rows[0]["concern_flags"] == ["alpha", "beta", "delta", "gamma", "zeta"]

    with (tmp_path / "evidence_table.csv").open("r", encoding="utf-8", newline="") as fh:
        csv_rows = list(csv.DictReader(fh))
    assert csv_rows[0]["concern_flags"] == "alpha;beta;delta;gamma;zeta"


def test_evidence_table_extracts_robustness_fields_by_step_id(tmp_path: Path) -> None:
    toi = "TOI-ROBUST.01"
    toi_dir = tmp_path / toi / "steps"
    model_raw_path = _write_step(toi_dir / "03_model_compete_raw.json", {"verdict": "SINUSOID_DOMINANT"})
    model_detrended_path = _write_step(
        toi_dir / "04_model_compete_detrended.json",
        {"result": {"verdict": "TRANSIT_PLUS_VARIABILITY"}},
    )
    fpp_raw_path = _write_step(toi_dir / "05_fpp_raw.json", {"result": {"fpp": 0.03}})
    fpp_detrended_path = _write_step(toi_dir / "06_fpp_detrended.json", {"fpp": 0.01})

    toi_result = {
        "toi": toi,
        "concern_flags": [],
        "steps": [
            {
                "step_id": "model_compete_raw",
                "op": "model_compete",
                "status": "ok",
                "step_output_path": model_raw_path,
            },
            {
                "step_id": "model_compete_detrended",
                "op": "model_compete",
                "status": "ok",
                "step_output_path": model_detrended_path,
            },
            {"step_id": "fpp_raw", "op": "fpp_run", "status": "ok", "step_output_path": fpp_raw_path},
            {
                "step_id": "fpp_detrended",
                "op": "fpp_run",
                "status": "ok",
                "step_output_path": fpp_detrended_path,
            },
        ],
    }

    rows = _write_evidence_table(out_dir=tmp_path, toi_results=[toi_result])
    row = rows[0]
    assert row["model_compete_raw_verdict"] == "SINUSOID_DOMINANT"
    assert row["model_compete_detrended_verdict"] == "TRANSIT_PLUS_VARIABILITY"
    assert row["fpp_raw"] == 0.03
    assert row["fpp_detrended"] == 0.01
    assert row["fpp_delta_detrended_minus_raw"] == pytest.approx(-0.02)
    assert row["robustness_recommended_variant"] == "detrended"
    assert row["detrend_invariance_policy_version"] == "v1"
    assert row["detrend_invariance_policy_verdict"] == "NON_INVARIANT"
    assert row["detrend_invariance_policy_reason_code"] == "MODEL_VERDICT_CHANGED"
    assert row["detrend_invariance_policy_fpp_delta_abs_threshold"] == pytest.approx(0.01)
    assert row["detrend_invariance_policy_observed_fpp_delta_abs"] == pytest.approx(0.02)
    assert row["detrend_invariance_policy_observed_model_verdict_changed"] is True


def test_evidence_table_prefers_fpp_run_for_final_fpp_value(tmp_path: Path) -> None:
    toi = "TOI-FPP-RUN.01"
    toi_dir = tmp_path / toi / "steps"
    legacy_fpp_path = _write_step(toi_dir / "01_fpp_legacy.json", {"fpp": 0.9})
    staged_fpp_run_path = _write_step(toi_dir / "02_fpp_run.json", {"result": {"fpp": 0.02}})

    toi_result = {
        "toi": toi,
        "concern_flags": [],
        "steps": [
            {"step_id": "fpp_legacy", "op": "fpp", "status": "ok", "step_output_path": legacy_fpp_path},
            {"step_id": "fpp", "op": "fpp_run", "status": "ok", "step_output_path": staged_fpp_run_path},
        ],
    }

    rows = _write_evidence_table(out_dir=tmp_path, toi_results=[toi_result])
    row = rows[0]
    assert row["fpp"] == 0.02


def test_evidence_table_detrend_invariance_policy_invariant_case(tmp_path: Path) -> None:
    toi = "TOI-ROBUST.02"
    toi_dir = tmp_path / toi / "steps"
    model_raw_path = _write_step(toi_dir / "03_model_compete_raw.json", {"verdict": "TRANSIT_PLUS_VARIABILITY"})
    model_detrended_path = _write_step(
        toi_dir / "04_model_compete_detrended.json",
        {"result": {"verdict": "TRANSIT_PLUS_VARIABILITY"}},
    )
    fpp_raw_path = _write_step(toi_dir / "05_fpp_raw.json", {"result": {"fpp": 0.03}})
    fpp_detrended_path = _write_step(toi_dir / "06_fpp_detrended.json", {"fpp": 0.031})

    toi_result = {
        "toi": toi,
        "concern_flags": [],
        "steps": [
            {
                "step_id": "model_compete_raw",
                "op": "model_compete",
                "status": "ok",
                "step_output_path": model_raw_path,
            },
            {
                "step_id": "model_compete_detrended",
                "op": "model_compete",
                "status": "ok",
                "step_output_path": model_detrended_path,
            },
            {"step_id": "fpp_raw", "op": "fpp_run", "status": "ok", "step_output_path": fpp_raw_path},
            {
                "step_id": "fpp_detrended",
                "op": "fpp_run",
                "status": "ok",
                "step_output_path": fpp_detrended_path,
            },
        ],
    }

    rows = _write_evidence_table(out_dir=tmp_path, toi_results=[toi_result])
    row = rows[0]
    assert row["detrend_invariance_policy_version"] == "v1"
    assert row["detrend_invariance_policy_verdict"] == "INVARIANT"
    assert row["detrend_invariance_policy_reason_code"] == "PASS"
    assert row["detrend_invariance_policy_fpp_delta_abs_threshold"] == pytest.approx(0.01)
    assert row["detrend_invariance_policy_observed_fpp_delta_abs"] == pytest.approx(0.001)
    assert row["detrend_invariance_policy_observed_model_verdict_changed"] is False
