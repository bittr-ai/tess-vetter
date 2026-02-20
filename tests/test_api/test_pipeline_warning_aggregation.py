from __future__ import annotations

import csv
import importlib
from pathlib import Path

import numpy as np
import pytest

from tess_vetter.features import FeatureConfig
from tess_vetter.pipeline import enrich_candidate
from tess_vetter.validation.result_schema import VettingBundleResult, ok_result


def _write_sector_csv(path: Path, *, time_btjd: np.ndarray, flux: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time_btjd", "flux", "flux_err", "quality"])
        writer.writeheader()
        for t, f in zip(time_btjd.tolist(), flux.tolist(), strict=True):
            writer.writerow({"time_btjd": t, "flux": f, "flux_err": 5e-4, "quality": 0})


def test_pipeline_adds_u14_u15_u17_warnings_deterministically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tic_id = 123
    period_days = 10.0
    t0_btjd = 100.0
    duration_hours = 2.0
    catalog_depth_ppm = 1000.0

    n = 1000
    sec1_t = np.linspace(95.0, 105.0, n, dtype=np.float64)
    sec2_t = np.linspace(105.0, 115.0, n, dtype=np.float64)

    sec1_f = np.ones_like(sec1_t)
    sec2_f = np.ones_like(sec2_t)

    half_dur_days = (duration_hours / 24.0) / 2.0
    sec1_f[np.abs(sec1_t - t0_btjd) <= half_dur_days] -= 500.0e-6
    sec2_f[np.abs(sec2_t - (t0_btjd + period_days)) <= half_dur_days] -= 500.0e-6

    _write_sector_csv(tmp_path / f"tic{tic_id}" / "sector1_pdcsap.csv", time_btjd=sec1_t, flux=sec1_f)
    _write_sector_csv(tmp_path / f"tic{tic_id}" / "sector2_pdcsap.csv", time_btjd=sec2_t, flux=sec2_f)

    def _fake_vet_candidate(*_args, **_kwargs) -> VettingBundleResult:
        results = [
            ok_result("V01", "Odd-Even Depth", metrics={"delta_ppm": 120.0}),
            ok_result("V02", "Secondary Eclipse", metrics={"secondary_depth_ppm": 110.0}),
            ok_result(
                "V07",
                "ExoFOP TOI Lookup",
                metrics={"found": True},
                raw={"found": True, "row": {"sectors": "2, 4"}},
            ),
        ]
        return VettingBundleResult(
            results=results,
            warnings=["existing_warning"],
            provenance={},
            inputs_summary={},
        )

    vet_mod = importlib.import_module("tess_vetter.api.vet")

    monkeypatch.setattr(vet_mod, "vet_candidate", _fake_vet_candidate)

    raw, row = enrich_candidate(
        tic_id,
        toi=None,
        period_days=period_days,
        t0_btjd=t0_btjd,
        duration_hours=duration_hours,
        depth_ppm=catalog_depth_ppm,
        config=FeatureConfig(
            bulk_mode=True,
            network_ok=False,
            local_data_path=str(tmp_path),
        ),
    )

    assert row["status"] == "OK"

    warnings = raw["provenance"]["warnings"]
    assert warnings[0] == "existing_warning"
    assert warnings[1] == "ExoFOP catalog sectors [2, 4] differ from discovered sectors [1, 2]"
    assert (
        warnings[2]
        == "Combined EB concern: V01 odd/even asymmetry and V02 secondary depth are each >=10% of primary depth"
    )
    assert warnings[3].startswith("Measured depth differs from catalog depth by 50.0% (>20%)")
    assert len(warnings) == 4
