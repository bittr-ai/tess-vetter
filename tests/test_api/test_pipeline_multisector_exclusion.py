from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from bittr_tess_vetter.api.enrichment import validate_enriched_row
from bittr_tess_vetter.features import FeatureConfig
from bittr_tess_vetter.pipeline import enrich_candidate


def _write_sector_csv(path: Path, *, time_btjd: np.ndarray) -> None:
    flux = np.ones_like(time_btjd)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time_btjd", "flux", "flux_err", "quality"])
        writer.writeheader()
        for t, f in zip(time_btjd.tolist(), flux.tolist(), strict=True):
            writer.writerow({"time_btjd": t, "flux": f, "flux_err": 5e-4, "quality": 0})


def test_multisector_drops_bad_time_coverage_sector(tmp_path: Path) -> None:
    tic_id = 123
    base = tmp_path / "data"

    # sector1 does not cover t0=100
    _write_sector_csv(base / f"tic{tic_id}" / "sector1_pdcsap.csv", time_btjd=np.linspace(0.0, 10.0, 300))
    # sector2 covers t0=100
    _write_sector_csv(
        base / f"tic{tic_id}" / "sector2_pdcsap.csv", time_btjd=np.linspace(90.0, 120.0, 300)
    )

    _raw, row = enrich_candidate(
        tic_id,
        toi=None,
        period_days=10.0,
        t0_btjd=100.0,
        duration_hours=2.0,
        depth_ppm=500.0,
        config=FeatureConfig(network_ok=False, bulk_mode=True, local_data_path=str(base)),
    )

    assert row["status"] == "OK"
    validate_enriched_row(row)
    assert row["inputs_summary"]["sectors"] == [2]
    assert isinstance(row["inputs_summary"].get("btv_code_hash"), str)
    assert row["inputs_summary"]["btv_code_hash"]
    dep = row["inputs_summary"].get("btv_dependency_versions")
    assert isinstance(dep, dict)
