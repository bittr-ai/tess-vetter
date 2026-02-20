from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tess_vetter.features import FeatureConfig
from tess_vetter.pipeline import enrich_worklist


def _write_sector_csv(path: Path, *, time_btjd: np.ndarray, flux: np.ndarray) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time_btjd", "flux", "flux_err", "quality"])
        writer.writeheader()
        for t, f in zip(time_btjd.tolist(), flux.tolist(), strict=True):
            writer.writerow({"time_btjd": t, "flux": f, "flux_err": 5e-4, "quality": 0})


def test_enrich_worklist_passes_sectors_and_records_selection(tmp_path: Path) -> None:
    # Create local data with two sectors but request only sector 2.
    tic_id = 123
    base = tmp_path / "data"
    n = 500
    time_btjd = np.linspace(90.0, 120.0, n, dtype=np.float64)
    flux = np.ones(n, dtype=np.float64)
    _write_sector_csv(base / f"tic{tic_id}" / "sector1_pdcsap.csv", time_btjd=time_btjd, flux=flux)
    _write_sector_csv(base / f"tic{tic_id}" / "sector2_pdcsap.csv", time_btjd=time_btjd, flux=flux)

    out = tmp_path / "out.jsonl"
    worklist = iter(
        [
            {
                "tic_id": tic_id,
                "toi": None,
                "period_days": 10.0,
                "t0_btjd": 100.0,
                "duration_hours": 2.0,
                "depth_ppm": 500.0,
                "sectors": [2],
            }
        ]
    )

    summary = enrich_worklist(
        worklist_iter=worklist,
        output_path=out,
        config=FeatureConfig(network_ok=False, bulk_mode=True, local_data_path=str(base)),
        resume=False,
        progress_interval=1,
    )

    assert summary.total_input == 1
    assert out.exists()
    row = json.loads(out.read_text().splitlines()[0])
    assert row["status"] == "OK"

    # Raw evidence isn't in JSONL output, but we record selection into inputs_summary or provenance in raw.
    # Since EnrichedRow carries inputs_summary, assert it includes the selected sector.
    inputs = row.get("inputs_summary", {})
    assert isinstance(inputs, dict)
    assert inputs.get("sectors") == [2]
