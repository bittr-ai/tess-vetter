from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from tess_vetter.cli.enrich_cli import cli
from tess_vetter.cli.progress_metadata import (
    build_single_candidate_progress,
    write_progress_metadata_atomic,
)


def test_btv_pipeline_run_composition_real_subprocess_resume_smoke(tmp_path: Path) -> None:
    """Run pipeline without subprocess monkeypatching using a local resume-skipped report step."""
    runner = CliRunner()
    toi = "TOI-SMOKE.01"
    out_dir = tmp_path / "pipeline_run_smoke_real"
    toi_dir = out_dir / toi
    steps_dir = toi_dir / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)

    output_path = steps_dir / "01_report_seed.json"
    progress_path = tmp_path / "report_seed.progress.json"

    candidate_meta = {
        "tic_id": 123456789,
        "period_days": 7.25,
        "t0_btjd": 2000.125,
        "duration_hours": 2.5,
    }

    output_path.write_text(
        json.dumps(
            {
                "schema_version": "cli.report.v2",
                "report": {
                    "summary": {
                        "tic_id": candidate_meta["tic_id"],
                        "ephemeris": {
                            "period_days": candidate_meta["period_days"],
                            "t0_btjd": candidate_meta["t0_btjd"],
                            "duration_hours": candidate_meta["duration_hours"],
                        },
                        "input_depth_ppm": 420.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    write_progress_metadata_atomic(
        progress_path,
        build_single_candidate_progress(
            command="report",
            candidate=candidate_meta,
            resume=True,
            status="completed",
            output_path=str(output_path),
            errors=0,
        ),
    )

    composition_path = tmp_path / "smoke.composition.json"
    composition_path.write_text(
        json.dumps(
            {
                "schema_version": "pipeline.composition.v1",
                "id": "smoke_report_resume",
                "description": "Local smoke run for pipeline subprocess path",
                "defaults": {},
                "steps": [
                    {
                        "id": "report_seed",
                        "op": "report",
                        "inputs": {
                            "tic_id": candidate_meta["tic_id"],
                            "period_days": candidate_meta["period_days"],
                            "t0_btjd": candidate_meta["t0_btjd"],
                            "duration_hours": candidate_meta["duration_hours"],
                            "depth_ppm": 420.0,
                            "resume": True,
                            "progress_path": str(progress_path),
                        },
                    }
                ],
                "final_mapping": {},
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        cli,
        [
            "pipeline",
            "run",
            "--composition-file",
            str(composition_path),
            "--toi",
            toi,
            "--out-dir",
            str(out_dir),
            "--no-network",
            "--max-workers",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Pipeline run complete: n_tois=1 ok=1 partial=0 failed=0" in result.output

    manifest = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["profile_id"] == "smoke_report_resume"
    assert manifest["counts"] == {"n_tois": 1, "n_ok": 1, "n_partial": 0, "n_failed": 0}

    toi_result = json.loads((toi_dir / "pipeline_result.json").read_text(encoding="utf-8"))
    assert toi_result["status"] == "ok"
    assert len(toi_result["steps"]) == 1
    assert toi_result["steps"][0]["step_id"] == "report_seed"
    assert toi_result["steps"][0]["status"] == "ok"
    assert toi_result["steps"][0].get("skipped_resume") is False
    assert int(toi_result["steps"][0].get("attempt") or 0) >= 1

    stderr_log = (toi_dir / "logs" / "report_seed.stderr.log").read_text(encoding="utf-8")
    assert isinstance(stderr_log, str)
