from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from bittr_tess_vetter.api.enrichment import validate_enriched_row
from bittr_tess_vetter.features import FeatureConfig
from bittr_tess_vetter.pipeline import enrich_candidate


def _write_sector_csv(path: Path, *, time_btjd: np.ndarray, flux: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time_btjd", "flux", "flux_err", "quality"])
        writer.writeheader()
        for t, f in zip(time_btjd.tolist(), flux.tolist(), strict=True):
            writer.writerow(
                {"time_btjd": t, "flux": f, "flux_err": 5e-4, "quality": 0},
            )


def test_enrich_candidate_offline_with_local_data_path(tmp_path: Path) -> None:
    tic_id = 123
    period_days = 10.0
    t0_btjd = 100.0
    duration_hours = 2.0
    depth_ppm = 500.0

    n = 2000
    time_btjd = np.linspace(90.0, 120.0, n, dtype=np.float64)
    duration_days = duration_hours / 24.0
    in_transit = np.abs(time_btjd - t0_btjd) < (duration_days / 2.0)

    rng = np.random.default_rng(0)
    flux = np.ones(n, dtype=np.float64) + rng.normal(0.0, 5e-4, size=n)
    flux[in_transit] -= depth_ppm * 1e-6

    _write_sector_csv(tmp_path / f"tic{tic_id}" / "sector1_pdcsap.csv", time_btjd=time_btjd, flux=flux)

    raw, row = enrich_candidate(
        tic_id,
        toi=None,
        period_days=period_days,
        t0_btjd=t0_btjd,
        duration_hours=duration_hours,
        depth_ppm=depth_ppm,
        config=FeatureConfig(
            bulk_mode=True,
            network_ok=False,
            local_data_path=str(tmp_path),
        ),
    )

    assert raw["target"]["tic_id"] == tic_id
    assert row["status"] == "OK"
    validate_enriched_row(row)

