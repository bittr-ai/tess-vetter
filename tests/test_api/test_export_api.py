from __future__ import annotations

import csv
import json
from io import StringIO

import numpy as np

from bittr_tess_vetter.api.export import export_bundle
from bittr_tess_vetter.api.types import Candidate, Ephemeris, LightCurve
from bittr_tess_vetter.api.vet import vet_candidate


def _bundle_one_check() -> tuple[LightCurve, Candidate]:
    time = np.linspace(0.0, 10.0, 500, dtype=np.float64)
    flux = np.ones_like(time)
    flux_err = np.full_like(time, 1e-3)
    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    cand = Candidate(ephemeris=Ephemeris(period_days=2.0, t0_btjd=0.5, duration_hours=2.0), depth_ppm=1000.0)
    return lc, cand


def test_export_bundle_json_strips_raw_by_default() -> None:
    lc, cand = _bundle_one_check()
    bundle = vet_candidate(lc, cand, checks=["V01"])

    out = export_bundle(bundle, format="json")
    assert out is not None
    payload = json.loads(out)
    assert payload["schema_version"] == 1
    results = payload["bundle"]["results"]
    assert isinstance(results, list) and results
    assert "raw" not in results[0]


def test_export_bundle_csv_smoke() -> None:
    lc, cand = _bundle_one_check()
    bundle = vet_candidate(lc, cand, checks=["V01"])

    out = export_bundle(bundle, format="csv")
    assert out is not None
    reader = csv.DictReader(StringIO(out))
    rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["id"] == "V01"
    assert "metrics_json" in rows[0]


def test_export_bundle_md_smoke() -> None:
    lc, cand = _bundle_one_check()
    bundle = vet_candidate(lc, cand, checks=["V01"])

    out = export_bundle(bundle, format="md", title="My Report")
    assert out is not None
    assert "My Report" in out

