from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from tess_vetter.features import FeatureConfig
from tess_vetter.pipeline import enrich_candidate


def _write_sector_csv(path: Path, *, time_btjd: np.ndarray, flux: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time_btjd", "flux", "flux_err", "quality"])
        writer.writeheader()
        for t, f in zip(time_btjd.tolist(), flux.tolist(), strict=True):
            writer.writerow({"time_btjd": t, "flux": f, "flux_err": 5e-4, "quality": 0})


def test_enrich_candidate_fails_when_t0_outside_observed_span(tmp_path: Path) -> None:
    base = tmp_path / "data"
    tic_id = 123
    time_btjd = np.linspace(90.0, 120.0, 1000, dtype=np.float64)
    flux = np.ones_like(time_btjd)
    _write_sector_csv(base / f"tic{tic_id}" / "sector1_pdcsap.csv", time_btjd=time_btjd, flux=flux)

    _raw, row = enrich_candidate(
        tic_id,
        toi=None,
        # Choose an ephemeris with no predicted transit epoch inside [90, 120].
        # The pipeline shifts t0 by integer periods to find an epoch near each sector.
        period_days=1000.0,
        t0_btjd=200.0,  # outside [90, 120] and too far to shift into range
        duration_hours=2.0,
        depth_ppm=500.0,
        config=FeatureConfig(network_ok=False, bulk_mode=True, local_data_path=str(base)),
    )
    assert row["status"] == "ERROR"
    assert row["error_class"] == "InsufficientTimeCoverageError"
