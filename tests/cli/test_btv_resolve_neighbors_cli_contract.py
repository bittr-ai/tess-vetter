from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from click.testing import CliRunner

from bittr_tess_vetter.cli.resolve_neighbors_cli import resolve_neighbors_command
from bittr_tess_vetter.platform.catalogs.models import SourceRecord


class _FakeSource:
    def __init__(
        self,
        *,
        source_id: int,
        ra: float,
        dec: float,
        phot_g_mean_mag: float | None = None,
        ruwe: float | None = None,
    ) -> None:
        self.source_id = source_id
        self.ra = ra
        self.dec = dec
        self.phot_g_mean_mag = phot_g_mean_mag
        self.ruwe = ruwe


class _FakeNeighbor(_FakeSource):
    def __init__(
        self,
        *,
        source_id: int,
        ra: float,
        dec: float,
        separation_arcsec: float,
        phot_g_mean_mag: float | None = None,
        delta_mag: float | None = None,
        ruwe: float | None = None,
    ) -> None:
        super().__init__(
            source_id=source_id,
            ra=ra,
            dec=dec,
            phot_g_mean_mag=phot_g_mean_mag,
            ruwe=ruwe,
        )
        self.separation_arcsec = separation_arcsec
        self.delta_mag = delta_mag


class _FakeGaiaResult:
    def __init__(self, *, source: _FakeSource | None, neighbors: list[_FakeNeighbor]) -> None:
        self.source = source
        self.neighbors = neighbors
        self.source_record = SourceRecord(
            name="gaia_dr3",
            version="dr3",
            retrieved_at=datetime.now(UTC),
            query="fake",
        )


def test_btv_resolve_neighbors_success_payload_contract(monkeypatch, tmp_path: Path) -> None:
    def _fake_query_gaia_by_position_sync(ra: float, dec: float, radius_arcsec: float) -> _FakeGaiaResult:
        _ = ra, dec, radius_arcsec
        return _FakeGaiaResult(
            source=_FakeSource(source_id=9001, ra=120.001, dec=-30.002, phot_g_mean_mag=10.0, ruwe=1.1),
            neighbors=[
                _FakeNeighbor(
                    source_id=9002,
                    ra=120.01,
                    dec=-30.01,
                    separation_arcsec=4.2,
                    phot_g_mean_mag=12.0,
                    delta_mag=2.0,
                    ruwe=1.0,
                ),
                _FakeNeighbor(
                    source_id=9003,
                    ra=120.02,
                    dec=-30.02,
                    separation_arcsec=8.4,
                    phot_g_mean_mag=13.0,
                    delta_mag=3.0,
                    ruwe=1.3,
                ),
            ],
        )

    monkeypatch.setattr(
        "bittr_tess_vetter.cli.resolve_neighbors_cli.query_gaia_by_position_sync",
        _fake_query_gaia_by_position_sync,
    )

    out_path = tmp_path / "reference_sources.json"
    runner = CliRunner()
    result = runner.invoke(
        resolve_neighbors_command,
        [
            "--tic-id",
            "123",
            "--ra-deg",
            "120.0",
            "--dec-deg",
            "-30.0",
            "--network-ok",
            "--max-neighbors",
            "1",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "reference_sources.v1"
    assert payload["target"]["tic_id"] == 123
    assert len(payload["reference_sources"]) == 2
    assert payload["reference_sources"][0]["name"] == "Target TIC 123"
    assert payload["reference_sources"][0]["role"] == "target"
    assert payload["reference_sources"][0]["tic_id"] == 123
    assert payload["reference_sources"][0]["meta"]["source"] == "gaia_dr3_primary"
    assert payload["reference_sources"][1]["source_id"] == "gaia:9002"
    assert payload["reference_sources"][1]["role"] == "companion"
    assert payload["reference_sources"][1]["separation_arcsec"] == 4.2
    assert payload["verdict"] == "NEIGHBORS_RESOLVED"
    assert payload["verdict_source"] == "$.provenance.gaia_resolution.n_neighbors_added"
    assert payload["result"]["verdict"] == payload["verdict"]
    assert payload["result"]["verdict_source"] == payload["verdict_source"]
    assert payload["provenance"]["gaia_resolution"]["n_neighbors_added"] == 1
    assert payload["multiplicity_risk"]["status"] == "LOW"
    assert payload["multiplicity_risk"]["reasons"] == ["NO_MULTIPLICITY_FLAGS"]


def test_btv_resolve_neighbors_gaia_error_falls_back_to_target_only(monkeypatch, tmp_path: Path) -> None:
    def _fake_query_gaia_by_position_sync(ra: float, dec: float, radius_arcsec: float) -> Any:
        _ = ra, dec, radius_arcsec
        raise RuntimeError("Gaia down")

    monkeypatch.setattr(
        "bittr_tess_vetter.cli.resolve_neighbors_cli.query_gaia_by_position_sync",
        _fake_query_gaia_by_position_sync,
    )

    out_path = tmp_path / "reference_sources_fallback.json"
    runner = CliRunner()
    result = runner.invoke(
        resolve_neighbors_command,
        [
            "--tic-id",
            "123",
            "--ra-deg",
            "120.0",
            "--dec-deg",
            "-30.0",
            "--network-ok",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "reference_sources.v1"
    assert len(payload["reference_sources"]) == 1
    assert payload["reference_sources"][0]["source_id"] == "tic:123"
    assert payload["verdict"] == "DATA_UNAVAILABLE"
    assert payload["verdict_source"] == "$.provenance.gaia_resolution.status"
    assert payload["result"]["verdict"] == payload["verdict"]
    assert payload["result"]["verdict_source"] == payload["verdict_source"]
    assert payload["provenance"]["gaia_resolution"]["status"] == "error_fallback_target_only"
    assert payload["multiplicity_risk"]["status"] == "UNKNOWN"
    assert "GAIA_UNAVAILABLE" in payload["multiplicity_risk"]["reasons"]


def test_btv_resolve_neighbors_emits_elevated_multiplicity_risk(monkeypatch, tmp_path: Path) -> None:
    class _FakeNssSource(_FakeSource):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.non_single_star = True
            self.duplicated_source = False

    def _fake_query_gaia_by_position_sync(ra: float, dec: float, radius_arcsec: float) -> _FakeGaiaResult:
        _ = ra, dec, radius_arcsec
        return _FakeGaiaResult(
            source=_FakeNssSource(
                source_id=9001,
                ra=120.001,
                dec=-30.002,
                phot_g_mean_mag=10.0,
                ruwe=1.8,
            ),
            neighbors=[],
        )

    monkeypatch.setattr(
        "bittr_tess_vetter.cli.resolve_neighbors_cli.query_gaia_by_position_sync",
        _fake_query_gaia_by_position_sync,
    )

    out_path = tmp_path / "reference_sources_multiplicity_risk.json"
    runner = CliRunner()
    result = runner.invoke(
        resolve_neighbors_command,
        [
            "--tic-id",
            "123",
            "--ra-deg",
            "120.0",
            "--dec-deg",
            "-30.0",
            "--network-ok",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    risk = payload["multiplicity_risk"]
    assert payload["verdict"] == "TARGET_ONLY"
    assert payload["verdict_source"] == "$.provenance.gaia_resolution.n_neighbors_added"
    assert payload["result"]["verdict"] == payload["verdict"]
    assert risk["status"] == "HIGH"
    assert "TARGET_NON_SINGLE_STAR" in risk["reasons"]
    assert "TARGET_RUWE_ELEVATED" in risk["reasons"]


def test_btv_resolve_neighbors_accepts_positional_toi_and_short_o(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def _fake_resolve_tic_id(*, tic_id: int | None, toi: str | None, network_ok: bool):
        seen["tic_id"] = tic_id
        seen["toi"] = toi
        seen["network_ok"] = network_ok
        return 123, None

    monkeypatch.setattr(
        "bittr_tess_vetter.cli.resolve_neighbors_cli._resolve_tic_id",
        _fake_resolve_tic_id,
    )

    out_path = tmp_path / "reference_sources_positional.json"
    runner = CliRunner()
    result = runner.invoke(
        resolve_neighbors_command,
        [
            "TOI-5807.01",
            "--ra-deg",
            "120.0",
            "--dec-deg",
            "-30.0",
            "--max-neighbors",
            "0",
            "-o",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["tic_id"] is None
    assert seen["toi"] == "TOI-5807.01"
    assert seen["network_ok"] is False
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "reference_sources.v1"
    assert payload["target"]["toi"] == "TOI-5807.01"
    assert payload["verdict"] == "DATA_UNAVAILABLE"
    assert payload["result"]["verdict"] == payload["verdict"]


def test_btv_resolve_neighbors_rejects_mismatched_positional_and_option_toi() -> None:
    runner = CliRunner()
    result = runner.invoke(
        resolve_neighbors_command,
        [
            "TOI-5807.01",
            "--toi",
            "TOI-4510.01",
            "--ra-deg",
            "120.0",
            "--dec-deg",
            "-30.0",
        ],
    )
    assert result.exit_code == 1
    assert "must match" in result.output


def test_btv_resolve_neighbors_report_file_inputs_override_candidate_flags(monkeypatch, tmp_path: Path) -> None:
    report_path = tmp_path / "resolve_neighbors.report.json"
    report_path.write_text(
        json.dumps(
            {
                "report": {
                    "summary": {
                        "tic_id": 555,
                        "ephemeris": {
                            "period_days": 6.0,
                            "t0_btjd": 2450.0,
                            "duration_hours": 2.0,
                        },
                        "input_depth_ppm": 400.0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "bittr_tess_vetter.cli.resolve_neighbors_cli._resolve_tic_id",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("should not resolve TOI with report file")),
    )

    def _fake_execute_resolve_neighbors(**kwargs: Any) -> dict[str, Any]:
        return {
            "schema_version": "reference_sources.v1",
            "reference_sources": [],
            "target": {"tic_id": kwargs["tic_id"], "toi": kwargs["toi"], "ra_deg": 0.0, "dec_deg": 0.0},
            "provenance": {"toi_resolution": kwargs["toi_resolution"]},
        }

    monkeypatch.setattr(
        "bittr_tess_vetter.cli.resolve_neighbors_cli._execute_resolve_neighbors",
        _fake_execute_resolve_neighbors,
    )

    out_path = tmp_path / "reference_sources_report_file.json"
    runner = CliRunner()
    result = runner.invoke(
        resolve_neighbors_command,
        [
            "--report-file",
            str(report_path),
            "--toi",
            "TOI-555.01",
            "--out",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Warning: --report-file provided; ignoring --tic-id/--toi" in result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["target"]["tic_id"] == 555
    assert payload["provenance"]["toi_resolution"]["source"] == "report_file"
