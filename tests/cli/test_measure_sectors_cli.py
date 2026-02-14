from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

import bittr_tess_vetter.cli.enrich_cli as enrich_cli
from bittr_tess_vetter.api.sector_metrics import SectorEphemerisMetrics


def test_measure_sectors_success_writes_json(monkeypatch, tmp_path: Path) -> None:
    def _fake_resolve_candidate_inputs(**_kwargs):
        return 123, 10.5, 2000.2, 2.5, None, {"source": "cli"}

    def _fake_execute_measure_sectors(**_kwargs):
        return {
            "schema_version": 1,
            "sector_measurements": [
                {"sector": 1, "depth_ppm": 500.0, "depth_err_ppm": 50.0, "quality_weight": 1.0}
            ],
            "provenance": {"command": "measure-sectors"},
        }

    monkeypatch.setattr(
        "bittr_tess_vetter.cli.measure_sectors_cli._resolve_candidate_inputs",
        _fake_resolve_candidate_inputs,
    )
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.measure_sectors_cli._execute_measure_sectors",
        _fake_execute_measure_sectors,
    )

    out_path = tmp_path / "sector_measurements.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "measure-sectors",
            "--tic-id",
            "123",
            "--period-days",
            "10.5",
            "--t0-btjd",
            "2000.2",
            "--duration-hours",
            "2.5",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert len(payload["sector_measurements"]) == 1


def test_measure_sectors_passes_vet_detrend_args(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_resolve_candidate_inputs(**_kwargs):
        return 123, 10.5, 2000.2, 2.5, None, {"source": "cli"}

    def _fake_execute_measure_sectors(**kwargs):
        captured.update(kwargs)
        return {
            "schema_version": 1,
            "sector_measurements": [],
            "provenance": {"command": "measure-sectors", "detrend": {"applied": True}},
        }

    monkeypatch.setattr(
        "bittr_tess_vetter.cli.measure_sectors_cli._resolve_candidate_inputs",
        _fake_resolve_candidate_inputs,
    )
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.measure_sectors_cli._execute_measure_sectors",
        _fake_execute_measure_sectors,
    )

    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "measure-sectors",
            "--tic-id",
            "123",
            "--period-days",
            "10.5",
            "--t0-btjd",
            "2000.2",
            "--duration-hours",
            "2.5",
            "--detrend",
            "transit_masked_bin_median",
            "--detrend-bin-hours",
            "8",
            "--detrend-buffer",
            "3",
            "--detrend-sigma-clip",
            "4",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["detrend"] == "transit_masked_bin_median"
    assert captured["detrend_bin_hours"] == 8.0
    assert captured["detrend_buffer"] == 3.0
    assert captured["detrend_sigma_clip"] == 4.0


def test_metric_to_measurement_uses_n_transits_for_quality_weight() -> None:
    from bittr_tess_vetter.cli.measure_sectors_cli import _metric_to_measurement

    m = SectorEphemerisMetrics(
        sector=10,
        n_total=100,
        n_valid=100,
        time_start_btjd=0.0,
        time_end_btjd=10.0,
        duration_days=10.0,
        cadence_seconds=120.0,
        n_in_transit=20,
        n_transits=0,
        n_out_of_transit=80,
        depth_hat_ppm=250.0,
        depth_sigma_ppm=30.0,
        score=5.0,
        flux_mad_ppm=100.0,
    )
    row = _metric_to_measurement(m, duration_hours=2.0)
    assert row["n_transits"] == 0
    assert row["quality_weight"] == 0.0
