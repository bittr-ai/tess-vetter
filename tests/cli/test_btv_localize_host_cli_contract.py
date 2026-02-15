from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from click.testing import CliRunner

import bittr_tess_vetter.cli.localize_host_cli as localize_host_cli
from bittr_tess_vetter.cli.localize_host_cli import localize_host_command
from bittr_tess_vetter.domain.lightcurve import LightCurveData
from bittr_tess_vetter.platform.io.mast_client import LightCurveNotFoundError


def _make_lc_data(*, tic_id: int, sector: int) -> LightCurveData:
    n = 16
    time = np.linspace(2000.0, 2001.0, n, dtype=np.float64)
    flux = np.ones(n, dtype=np.float64)
    flux_err = np.full(n, 1e-4, dtype=np.float64)
    quality = np.zeros(n, dtype=np.int32)
    valid = np.ones(n, dtype=bool)
    return LightCurveData(
        time=time,
        flux=flux,
        flux_err=flux_err,
        quality=quality,
        valid_mask=valid,
        tic_id=int(tic_id),
        sector=int(sector),
        cadence_seconds=120.0,
        provenance=None,
    )


def test_btv_localize_host_success_writes_contract_json(monkeypatch, tmp_path: Path) -> None:
    def _fake_resolve_candidate_inputs(**_kwargs: Any):
        return 123, 10.5, 2000.2, 2.5, None, {"source": "cli", "inputs": {"tic_id": 123}}

    def _fake_execute_localize_host(**_kwargs: Any) -> dict[str, Any]:
        return {
            "schema_version": "cli.localize_host.v1",
            "result": {"consensus_label": "ON_TARGET"},
            "inputs_summary": {"input_resolution": {"source": "cli", "inputs": {"tic_id": 123}}},
            "provenance": {
                "selected_sectors": [14],
                "requested_sectors": None,
                "tpf_sector_strategy": "best",
                "network_ok": False,
            },
        }

    monkeypatch.setattr(
        "bittr_tess_vetter.cli.localize_host_cli._resolve_candidate_inputs",
        _fake_resolve_candidate_inputs,
    )
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.localize_host_cli._execute_localize_host",
        _fake_execute_localize_host,
    )

    out_path = tmp_path / "localize_host.json"
    runner = CliRunner()
    result = runner.invoke(
        localize_host_command,
        [
            "--tic-id",
            "123",
            "--period-days",
            "10.5",
            "--t0-btjd",
            "2000.2",
            "--duration-hours",
            "2.5",
            "--random-seed",
            "17",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "cli.localize_host.v1"
    assert payload["result"]["consensus_label"] == "ON_TARGET"
    assert payload["inputs_summary"]["input_resolution"]["source"] == "cli"
    assert payload["provenance"]["selected_sectors"] == [14]
    assert payload["provenance"]["requested_sectors"] is None
    assert payload["provenance"]["tpf_sector_strategy"] == "best"
    assert payload["provenance"]["network_ok"] is False


def test_btv_localize_host_data_unavailable_when_tpf_missing_exits_4(monkeypatch) -> None:
    def _fake_resolve_candidate_inputs(**_kwargs: Any):
        return 123, 10.5, 2000.2, 2.5, None, {"source": "cli"}

    def _missing_tpf(**_kwargs: Any) -> dict[str, Any]:
        raise LightCurveNotFoundError("TPF unavailable for TIC 123")

    monkeypatch.setattr(
        "bittr_tess_vetter.cli.localize_host_cli._resolve_candidate_inputs",
        _fake_resolve_candidate_inputs,
    )
    monkeypatch.setattr("bittr_tess_vetter.cli.localize_host_cli._execute_localize_host", _missing_tpf)

    runner = CliRunner()
    result = runner.invoke(
        localize_host_command,
        [
            "--tic-id",
            "123",
            "--period-days",
            "10.5",
            "--t0-btjd",
            "2000.2",
            "--duration-hours",
            "2.5",
        ],
    )

    assert result.exit_code == 4


def test_btv_localize_host_strategy_requested_without_tpf_sector_exits_1() -> None:
    runner = CliRunner()
    result = runner.invoke(
        localize_host_command,
        [
            "--tic-id",
            "123",
            "--period-days",
            "10.5",
            "--t0-btjd",
            "2000.2",
            "--duration-hours",
            "2.5",
            "--tpf-sector-strategy",
            "requested",
        ],
    )

    assert result.exit_code == 1
    assert "requires at least one --tpf-sector" in result.output


def test_btv_localize_host_no_network_requires_explicit_sectors(monkeypatch) -> None:
    def _fake_resolve_candidate_inputs(**_kwargs: Any):
        return 123, 10.5, 2000.2, 2.5, None, {"source": "cli"}

    monkeypatch.setattr(
        "bittr_tess_vetter.cli.localize_host_cli._resolve_candidate_inputs",
        _fake_resolve_candidate_inputs,
    )

    runner = CliRunner()
    result = runner.invoke(
        localize_host_command,
        [
            "--tic-id",
            "123",
            "--period-days",
            "10.5",
            "--t0-btjd",
            "2000.2",
            "--duration-hours",
            "2.5",
            "--no-network",
        ],
    )

    assert result.exit_code == 1
    assert "--no-network requires explicit --sectors" in result.output


def test_execute_localize_host_uses_cache_only_when_no_network(monkeypatch) -> None:
    class _FakeWCS:
        def to_header(self, relax: bool = True):
            _ = relax
            return {"RA_OBJ": 120.123, "DEC_OBJ": -21.456}

    class _FakeMASTClient:
        def download_all_sectors(self, *_args: Any, **_kwargs: Any):
            raise AssertionError("download_all_sectors should not be called when network_ok=False")

        def download_lightcurve_cached(self, tic_id: int, sector: int, flux_type: str):
            _ = flux_type
            return _make_lc_data(tic_id=tic_id, sector=sector)

        def download_tpf_cached(self, tic_id: int, sector: int):
            _ = tic_id, sector
            n = 16
            time = np.linspace(2000.0, 2001.0, n, dtype=np.float64)
            flux = np.ones((n, 3, 3), dtype=np.float64)
            flux_err = np.full((n, 3, 3), 1e-3, dtype=np.float64)
            aperture = np.ones((3, 3), dtype=np.int32)
            quality = np.zeros(n, dtype=np.int32)
            return time, flux, flux_err, _FakeWCS(), aperture, quality

    monkeypatch.setattr(localize_host_cli, "MASTClient", _FakeMASTClient)
    monkeypatch.setattr(localize_host_cli, "_select_tpf_sectors", lambda **_kwargs: [14])
    monkeypatch.setattr(
        localize_host_cli,
        "localize_transit_host_multi_sector",
        lambda **_kwargs: {"per_sector_results": [], "consensus": {"consensus_label": "AMBIGUOUS"}},
    )

    payload = localize_host_cli._execute_localize_host(
        tic_id=123,
        period_days=10.5,
        t0_btjd=2000.2,
        duration_hours=2.5,
        ra_deg=None,
        dec_deg=None,
        network_ok=False,
        sectors=[14],
        tpf_sector_strategy="requested",
        tpf_sectors=[14],
        oot_margin_mult=1.5,
        oot_window_mult=10.0,
        centroid_method="centroid",
        prf_backend="prf_lite",
        baseline_shift_threshold=0.5,
        random_seed=42,
        input_resolution={"source": "cli"},
    )

    assert payload["schema_version"] == "cli.localize_host.v1"
    assert payload["provenance"]["network_ok"] is False
    assert payload["provenance"]["selected_sectors"] == [14]
