from __future__ import annotations

import csv
import json
from pathlib import Path

from bittr_tess_vetter.pipeline_composition.executor import _write_evidence_table


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
    localize_path = _write_step(
        toi_dir / "05_localize.json",
        {"result": {"consensus": {"action_hint": "collect_more_imaging"}}},
    )
    dilution_path = _write_step(toi_dir / "06_dilution.json", {"result": {"n_plausible_scenarios": 4}})
    fpp_path = _write_step(toi_dir / "07_fpp.json", {"result": {"fpp": 0.017}})

    toi_result = {
        "toi": toi,
        "concern_flags": [],
        "steps": [
            {"op": "model_compete", "status": "ok", "step_output_path": model_path},
            {"op": "systematics_proxy", "status": "ok", "step_output_path": systematics_path},
            {"op": "ephemeris_reliability", "status": "ok", "step_output_path": ephemeris_path},
            {"op": "timing", "status": "ok", "step_output_path": timing_path},
            {"op": "localize_host", "status": "ok", "step_output_path": localize_path},
            {"op": "dilution", "status": "ok", "step_output_path": dilution_path},
            {"op": "fpp", "status": "ok", "step_output_path": fpp_path},
        ],
    }

    rows = _write_evidence_table(out_dir=tmp_path, toi_results=[toi_result])
    row = rows[0]
    assert row["model_compete_verdict"] == "MODEL_OK"
    assert row["systematics_verdict"] == "SYSTEMATICS_OK"
    assert row["ephemeris_verdict"] == "EPHEMERIS_OK"
    assert row["timing_verdict"] == "TIMING_OK"
    assert row["localize_host_action_hint"] == "collect_more_imaging"
    assert row["dilution_n_plausible_scenarios"] == 4
    assert row["fpp"] == 0.017


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

