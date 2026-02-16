from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from click.testing import CliRunner

import bittr_tess_vetter.cli.enrich_cli as enrich_cli
from bittr_tess_vetter.api.sector_metrics import SectorEphemerisMetrics
from bittr_tess_vetter.cli import measure_sectors_cli


def _payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "sector_measurements": [
            {"sector": 1, "depth_ppm": 500.0, "depth_err_ppm": 50.0, "quality_weight": 1.0}
        ],
        "consistency": {
            "chi2": 0.0,
            "chi2_dof": 0,
            "chi2_pvalue": 1.0,
            "verdict": "UNRESOLVABLE",
            "depth_median_ppm": 500.0,
            "depth_std_ppm": 0.0,
            "outlier_sectors": [],
            "outlier_criterion": "|depth - weighted_mean| / depth_err_ppm > 3.0",
        },
        "recommended_sectors": [1],
        "recommended_sector_criterion": "test",
        "provenance": {"command": "measure-sectors"},
    }


def test_measure_sectors_accepts_positional_toi_and_short_o(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def _fake_resolve_candidate_inputs(**kwargs: Any):
        seen.update(kwargs)
        return 123, 10.5, 2000.2, 2.5, None, {"source": "toi_catalog"}

    monkeypatch.setattr(
        "bittr_tess_vetter.cli.measure_sectors_cli._resolve_candidate_inputs",
        _fake_resolve_candidate_inputs,
    )
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.measure_sectors_cli._execute_measure_sectors",
        lambda **_kwargs: _payload(),
    )

    out_path = tmp_path / "sector_measurements_positional.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        ["measure-sectors", "TOI-5807.01", "-o", str(out_path)],
    )

    assert result.exit_code == 0, result.output
    assert seen["toi"] == "TOI-5807.01"
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1


def test_measure_sectors_rejects_mismatched_positional_and_option_toi() -> None:
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        ["measure-sectors", "TOI-5807.01", "--toi", "TOI-4510.01"],
    )
    assert result.exit_code == 1
    assert "must match" in result.output


def test_measure_sectors_report_file_inputs_override_toi_and_seed_sectors(monkeypatch, tmp_path: Path) -> None:
    report_path = tmp_path / "measure.report.json"
    report_path.write_text(
        json.dumps(
            {
                "report": {
                    "summary": {
                        "tic_id": 987,
                        "ephemeris": {
                            "period_days": 3.2,
                            "t0_btjd": 2100.0,
                            "duration_hours": 1.9,
                        },
                        "input_depth_ppm": 220.0,
                    },
                    "provenance": {"sectors_used": [40, 41]},
                }
            }
        ),
        encoding="utf-8",
    )

    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.measure_sectors_cli._resolve_candidate_inputs",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("should not resolve TOI with report file")),
    )

    def _fake_execute_measure_sectors(**kwargs: Any):
        captured.update(kwargs)
        return _payload()

    monkeypatch.setattr(
        "bittr_tess_vetter.cli.measure_sectors_cli._execute_measure_sectors",
        _fake_execute_measure_sectors,
    )

    out_path = tmp_path / "measure_from_report.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "measure-sectors",
            "--report-file",
            str(report_path),
            "--toi",
            "TOI-987.01",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Warning: --report-file provided; ignoring --toi" in result.output
    assert captured["tic_id"] == 987
    assert captured["period_days"] == 3.2
    assert captured["t0_btjd"] == 2100.0
    assert captured["duration_hours"] == 1.9
    assert captured["sectors"] == [40, 41]
    assert captured["sectors_explicit"] is False
    assert captured["sector_selection_source"] == "report_file"
    assert captured["report_file_path"] == str(report_path.resolve())


def test_measure_sectors_resume_skips_existing_completed_output(monkeypatch, tmp_path: Path) -> None:
    out_path = tmp_path / "measure_resume.json"
    out_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sector_measurements": [],
                "provenance": {"command": "measure-sectors"},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "bittr_tess_vetter.cli.measure_sectors_cli._execute_measure_sectors",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("should skip due to resume")),
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
            "--resume",
            "--out",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Skipped resume: existing completed output" in result.output


def test_measure_sectors_enrichment_fields_and_recommended_sectors() -> None:
    rows = [
        {
            "sector": 10,
            "depth_ppm": 500.0,
            "depth_err_ppm": 20.0,
            "duration_hours": 2.0,
            "duration_err_hours": 0.0,
            "n_transits": 3,
            "shape_metric": 1.0,
            "quality_weight": 1.0,
        },
        {
            "sector": 11,
            "depth_ppm": 510.0,
            "depth_err_ppm": 20.0,
            "duration_hours": 2.0,
            "duration_err_hours": 0.0,
            "n_transits": 2,
            "shape_metric": 1.1,
            "quality_weight": 1.0,
        },
        {
            "sector": 12,
            "depth_ppm": 700.0,
            "depth_err_ppm": 30.0,
            "duration_hours": 2.0,
            "duration_err_hours": 0.0,
            "n_transits": 2,
            "shape_metric": 0.9,
            "quality_weight": 1.0,
        },
    ]
    consistency = measure_sectors_cli._build_consistency_enrichment(rows)
    assert set(consistency) == {
        "chi2",
        "chi2_dof",
        "chi2_pvalue",
        "verdict",
        "depth_median_ppm",
        "depth_std_ppm",
        "outlier_sectors",
        "outlier_criterion",
    }
    assert consistency["chi2_dof"] == 2
    assert consistency["verdict"] == "INCONSISTENT"
    assert consistency["outlier_sectors"] == [12]
    recommended = measure_sectors_cli._recommended_sectors(rows)
    assert recommended == [10, 11]


def test_metric_to_measurement_uses_n_transits_for_quality_weight() -> None:
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
    row = measure_sectors_cli._metric_to_measurement(m, duration_hours=2.0)
    assert row["n_transits"] == 0
    assert row["quality_weight"] == 0.0
