from __future__ import annotations

import csv
import functools
from pathlib import Path

import numpy as np
import pytest

from tess_vetter.api.types import Candidate, Ephemeris, LightCurve
from tess_vetter.report import build_report

_TOI_DATA_DIR = Path("docs/tutorials/data/tic188646744")
_TOI_SECTORS = (55, 75, 82, 83)


@functools.lru_cache(maxsize=1)
def _load_toi5807_lightcurve() -> LightCurve:
    time_all: list[np.ndarray] = []
    flux_all: list[np.ndarray] = []
    flux_err_all: list[np.ndarray] = []

    for sector in _TOI_SECTORS:
        path = _TOI_DATA_DIR / f"sector{sector}_pdcsap.csv"
        with path.open(newline="") as f:
            for line in f:
                if not line.startswith("#"):
                    header = line
                    break
            else:
                raise ValueError(f"Missing CSV header in {path}")

            reader = csv.DictReader([header] + f.readlines())
            rows = list(reader)

        time = np.asarray([float(r["time_btjd"]) for r in rows], dtype=np.float64)
        flux = np.asarray([float(r["flux"]) for r in rows], dtype=np.float64)
        flux_err = np.asarray([float(r["flux_err"]) for r in rows], dtype=np.float64)
        quality = np.asarray([int(r["quality"]) for r in rows], dtype=np.int32)
        valid = quality == 0
        time_all.append(time[valid])
        flux_all.append(flux[valid])
        flux_err_all.append(flux_err[valid])

    time_cat = np.concatenate(time_all)
    flux_cat = np.concatenate(flux_all)
    flux_err_cat = np.concatenate(flux_err_all)
    order = np.argsort(time_cat)
    return LightCurve(
        time=time_cat[order],
        flux=flux_cat[order],
        flux_err=flux_err_cat[order],
    )


@pytest.mark.slow
def test_toi5807_offline_scalar_benchmark_gate() -> None:
    required = [_TOI_DATA_DIR / f"sector{s}_pdcsap.csv" for s in _TOI_SECTORS]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        pytest.skip(f"TOI-5807 tutorial data missing: {missing}")

    lc = _load_toi5807_lightcurve()
    candidate = Candidate(
        ephemeris=Ephemeris(period_days=13.0177, t0_btjd=3401.2142, duration_hours=3.42),
        depth_ppm=253.0,
    )

    summary = build_report(lc, candidate, include_additional_plots=False).to_json()["summary"]
    assert int(summary["lc_summary"]["n_valid"]) > 50_000
    assert summary["lc_summary"]["snr"] is not None
    assert float(summary["lc_summary"]["snr"]) > 1.0

    odd_even = summary["odd_even_summary"]
    assert odd_even["depth_diff_sigma"] is not None
    assert abs(float(odd_even["depth_diff_sigma"])) < 5.0

    timing = summary["timing_summary"]
    assert int(timing["n_epochs_measured"]) >= 0
    if int(timing["n_epochs_measured"]) > 0:
        assert int(timing["outlier_count"]) <= int(timing["n_epochs_measured"])

    robustness = summary["lc_robustness_summary"]
    assert robustness["phase_0p5_bin_depth_ppm"] is not None
    assert robustness["secondary_depth_sigma"] is not None
