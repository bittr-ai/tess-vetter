from __future__ import annotations

import json
from pathlib import Path

from bittr_tess_vetter.features import FeatureConfig
from bittr_tess_vetter.pipeline import enrich_worklist


def test_enrich_worklist_writes_progress_manifest(tmp_path: Path) -> None:
    output = tmp_path / "out.jsonl"

    worklist = iter(
        [
            {
                "tic_id": 1,
                "toi": None,
                "period_days": 10.0,
                "t0_btjd": 100.0,
                "duration_hours": 2.0,
                "depth_ppm": 500.0,
            },
            {
                "tic_id": 2,
                "toi": None,
                "period_days": 10.0,
                "t0_btjd": 100.0,
                "duration_hours": 2.0,
                "depth_ppm": 500.0,
            },
        ]
    )

    summary = enrich_worklist(
        worklist_iter=worklist,
        output_path=output,
        config=FeatureConfig(network_ok=False, bulk_mode=True),
        resume=False,
        progress_interval=1,
    )

    assert summary.total_input == 2
    progress_path = output.with_suffix(output.suffix + ".progress.json")
    assert progress_path.exists()
    payload = json.loads(progress_path.read_text())
    assert payload["output_path"] == str(output)
    assert payload["total_input"] == 2
    assert payload["errors"] == 2  # offline without local_data_path
    assert payload["processed"] == 0


def test_enrich_worklist_resume_skips_existing(tmp_path: Path) -> None:
    output = tmp_path / "out.jsonl"

    row = {
        "tic_id": 1,
        "toi": None,
        "period_days": 10.0,
        "t0_btjd": 100.0,
        "duration_hours": 2.0,
        "depth_ppm": 500.0,
    }

    # First run writes one ERROR row.
    enrich_worklist(
        worklist_iter=iter([row]),
        output_path=output,
        config=FeatureConfig(network_ok=False, bulk_mode=True),
        resume=False,
        progress_interval=1,
    )

    # Second run should skip the existing candidate key when resume=True.
    summary = enrich_worklist(
        worklist_iter=iter([row]),
        output_path=output,
        config=FeatureConfig(network_ok=False, bulk_mode=True),
        resume=True,
        progress_interval=1,
    )

    assert summary.total_input == 1
    assert summary.skipped_resume == 1
    assert output.read_text().count("\n") == 1

