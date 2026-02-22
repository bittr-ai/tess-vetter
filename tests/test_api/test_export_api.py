from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path

import numpy as np

from tess_vetter.api.export import ExportFormatEnum, export_bundle
from tess_vetter.api.types import Candidate, Ephemeris, LightCurve
from tess_vetter.api.vet import vet_candidate
from tess_vetter.validation.result_schema import CheckResult, VettingBundleResult


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


def test_export_bundle_accepts_enum_format_values() -> None:
    lc, cand = _bundle_one_check()
    bundle = vet_candidate(lc, cand, checks=["V01"])

    out = export_bundle(bundle, format=ExportFormatEnum.JSON)
    assert out is not None
    payload = json.loads(out)
    assert payload["schema_version"] == 1


def test_export_bundle_uppercase_format_still_accepted() -> None:
    lc, cand = _bundle_one_check()
    bundle = vet_candidate(lc, cand, checks=["V01"])

    out = export_bundle(bundle, format="JSON")  # type: ignore[arg-type]
    assert out is not None
    payload = json.loads(out)
    assert payload["schema_version"] == 1


def test_export_bundle_json_schema_contract() -> None:
    lc, cand = _bundle_one_check()
    bundle = vet_candidate(lc, cand, checks=["V01"])

    out = export_bundle(bundle, format="json")
    assert out is not None
    payload = json.loads(out)
    assert set(payload.keys()) == {"schema_version", "bundle"}
    assert payload["schema_version"] == 1

    bundle_payload = payload["bundle"]
    assert set(bundle_payload.keys()) == {"inputs_summary", "provenance", "results", "warnings"}

    result_payload = bundle_payload["results"][0]
    assert set(result_payload.keys()) == {"confidence", "flags", "id", "metrics", "name", "notes", "provenance", "status"}


def test_export_bundle_csv_json_columns_are_valid_json() -> None:
    lc, cand = _bundle_one_check()
    bundle = vet_candidate(lc, cand, checks=["V01"])

    out = export_bundle(bundle, format="csv")
    assert out is not None
    row = next(csv.DictReader(StringIO(out)))
    assert row is not None

    for field in ("metrics_json", "flags_json", "notes_json", "provenance_json"):
        assert field in row
        json.loads(row[field])


def test_export_bundle_csv_include_raw_serializes_non_json_values() -> None:
    bundle = VettingBundleResult(
        results=[
            CheckResult(
                id="VXX",
                name="Synthetic",
                status="ok",
                confidence=1.0,
                metrics={"snr": 10.0},
                flags=[],
                notes=[],
                provenance={"source": "test"},
                raw={"path": Path("/tmp/raw"), "nested": {"value": 1}},
            )
        ],
        warnings=[],
        provenance={},
        inputs_summary={},
    )

    out = export_bundle(bundle, format="csv", include_raw=True)
    assert out is not None
    row = next(csv.DictReader(StringIO(out)))
    assert row is not None
    assert "raw_json" in row
    raw_payload = json.loads(row["raw_json"])
    assert raw_payload["path"] == "/tmp/raw"
    assert raw_payload["nested"] == {"value": 1}


def test_export_bundle_json_write_to_path_returns_none(tmp_path: Path) -> None:
    lc, cand = _bundle_one_check()
    bundle = vet_candidate(lc, cand, checks=["V01"])
    out_path = tmp_path / "export.json"

    out = export_bundle(bundle, format="json", path=out_path)
    assert out is None
    payload = json.loads(out_path.read_text())
    assert payload["schema_version"] == 1
