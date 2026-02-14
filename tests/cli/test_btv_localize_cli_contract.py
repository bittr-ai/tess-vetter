from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from click.testing import CliRunner

from bittr_tess_vetter.cli.localize_cli import localize_command
from bittr_tess_vetter.domain.lightcurve import LightCurveData
from bittr_tess_vetter.platform.catalogs.toi_resolution import LookupStatus
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


def test_btv_localize_success_writes_contract_json(monkeypatch, tmp_path: Path) -> None:
    def _fake_resolve_candidate_inputs(**_kwargs: Any):
        return 123, 10.5, 2000.2, 2.5, None, {"source": "cli", "inputs": {"tic_id": 123}}

    class _FakeWCS:
        def to_header(self, relax: bool = True):
            _ = relax
            return {"RA_OBJ": 120.123, "DEC_OBJ": -21.456}

    class _FakeMASTClient:
        def download_all_sectors(self, tic_id: int, flux_type: str, sectors: list[int] | None):
            _ = flux_type, sectors
            return [_make_lc_data(tic_id=tic_id, sector=14)]

        def download_tpf_cached(self, tic_id: int, sector: int):
            _ = tic_id, sector
            n = 16
            time = np.linspace(2000.0, 2001.0, n, dtype=np.float64)
            flux = np.ones((n, 3, 3), dtype=np.float64)
            flux_err = np.full((n, 3, 3), 1e-3, dtype=np.float64)
            aperture = np.ones((3, 3), dtype=np.int32)
            quality = np.zeros(n, dtype=np.int32)
            return time, flux, flux_err, _FakeWCS(), aperture, quality

    class _FakeLocalizationResult:
        def to_dict(self) -> dict[str, Any]:
            return {"verdict": "ON_TARGET", "bootstrap_seed": 17}

    monkeypatch.setattr(
        "bittr_tess_vetter.cli.localize_cli._resolve_candidate_inputs",
        _fake_resolve_candidate_inputs,
    )
    monkeypatch.setattr("bittr_tess_vetter.cli.localize_cli.MASTClient", _FakeMASTClient)
    monkeypatch.setattr("bittr_tess_vetter.cli.localize_cli._select_tpf_sectors", lambda **_kwargs: [14])
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.localize_cli.localize_transit_source",
        lambda **_kwargs: _FakeLocalizationResult(),
    )

    out_path = tmp_path / "localize.json"
    runner = CliRunner()
    result = runner.invoke(
        localize_command,
        [
            "--tic-id",
            "123",
            "--period-days",
            "10.5",
            "--t0-btjd",
            "2000.2",
            "--duration-hours",
            "2.5",
            "--bootstrap-seed",
            "17",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "cli.localize.v1"
    assert payload["result"]["verdict"] == "ON_TARGET"
    assert payload["inputs_summary"]["input_resolution"]["source"] == "cli"
    assert payload["provenance"]["selected_sector"] == 14
    assert payload["provenance"]["requested_sectors"] is None
    assert payload["provenance"]["tpf_sector_strategy"] == "best"
    assert payload["provenance"]["network_ok"] is False
    assert payload["provenance"]["coordinate_source"] == "tpf_meta"


def test_btv_localize_data_unavailable_when_tpf_missing_exits_4(monkeypatch) -> None:
    def _fake_resolve_candidate_inputs(**_kwargs: Any):
        return 123, 10.5, 2000.2, 2.5, None, {"source": "cli"}

    class _FakeMASTClient:
        def download_all_sectors(self, tic_id: int, flux_type: str, sectors: list[int] | None):
            _ = flux_type, sectors
            return [_make_lc_data(tic_id=tic_id, sector=14)]

        def download_tpf_cached(self, tic_id: int, sector: int):
            _ = tic_id, sector
            raise LightCurveNotFoundError("no tpf")

    monkeypatch.setattr(
        "bittr_tess_vetter.cli.localize_cli._resolve_candidate_inputs",
        _fake_resolve_candidate_inputs,
    )
    monkeypatch.setattr("bittr_tess_vetter.cli.localize_cli.MASTClient", _FakeMASTClient)
    monkeypatch.setattr("bittr_tess_vetter.cli.localize_cli._select_tpf_sectors", lambda **_kwargs: [14])

    runner = CliRunner()
    result = runner.invoke(
        localize_command,
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


def test_btv_localize_strategy_requested_without_tpf_sector_exits_1() -> None:
    runner = CliRunner()
    result = runner.invoke(
        localize_command,
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


def test_btv_localize_tic_lookup_fallback_success(monkeypatch, tmp_path: Path) -> None:
    def _fake_resolve_candidate_inputs(**_kwargs: Any):
        return 123, 10.5, 2000.2, 2.5, None, {"source": "cli"}

    class _FakeWCS:
        def to_header(self, relax: bool = True):
            _ = relax
            return {}

    class _FakeMASTClient:
        def download_all_sectors(self, tic_id: int, flux_type: str, sectors: list[int] | None):
            _ = flux_type, sectors
            return [_make_lc_data(tic_id=tic_id, sector=14)]

        def download_tpf_cached(self, tic_id: int, sector: int):
            _ = tic_id, sector
            n = 16
            time = np.linspace(2000.0, 2001.0, n, dtype=np.float64)
            flux = np.ones((n, 3, 3), dtype=np.float64)
            flux_err = np.full((n, 3, 3), 1e-3, dtype=np.float64)
            aperture = np.ones((3, 3), dtype=np.int32)
            quality = np.zeros(n, dtype=np.int32)
            return time, flux, flux_err, _FakeWCS(), aperture, quality

    class _LookupResult:
        status = LookupStatus.OK
        ra_deg = 11.25
        dec_deg = -3.5
        message = None

    class _FakeLocalizationResult:
        def to_dict(self) -> dict[str, Any]:
            return {"verdict": "ON_TARGET"}

    monkeypatch.setattr(
        "bittr_tess_vetter.cli.localize_cli._resolve_candidate_inputs",
        _fake_resolve_candidate_inputs,
    )
    monkeypatch.setattr("bittr_tess_vetter.cli.localize_cli.MASTClient", _FakeMASTClient)
    monkeypatch.setattr("bittr_tess_vetter.cli.localize_cli._select_tpf_sectors", lambda **_kwargs: [14])
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.localize_cli.lookup_tic_coordinates",
        lambda tic_id: _LookupResult(),
    )
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.localize_cli.localize_transit_source",
        lambda **_kwargs: _FakeLocalizationResult(),
    )

    out_path = tmp_path / "localize_lookup.json"
    runner = CliRunner()
    result = runner.invoke(
        localize_command,
        [
            "--tic-id",
            "123",
            "--period-days",
            "10.5",
            "--t0-btjd",
            "2000.2",
            "--duration-hours",
            "2.5",
            "--network-ok",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["provenance"]["coordinate_source"] == "tic_lookup"


def test_btv_localize_missing_coordinates_exits_1_when_lookup_unavailable(monkeypatch) -> None:
    def _fake_resolve_candidate_inputs(**_kwargs: Any):
        return 123, 10.5, 2000.2, 2.5, None, {"source": "cli"}

    class _FakeWCS:
        def to_header(self, relax: bool = True):
            _ = relax
            return {}

    class _FakeMASTClient:
        def download_all_sectors(self, tic_id: int, flux_type: str, sectors: list[int] | None):
            _ = flux_type, sectors
            return [_make_lc_data(tic_id=tic_id, sector=14)]

        def download_tpf_cached(self, tic_id: int, sector: int):
            _ = tic_id, sector
            n = 16
            time = np.linspace(2000.0, 2001.0, n, dtype=np.float64)
            flux = np.ones((n, 3, 3), dtype=np.float64)
            flux_err = np.full((n, 3, 3), 1e-3, dtype=np.float64)
            aperture = np.ones((3, 3), dtype=np.int32)
            quality = np.zeros(n, dtype=np.int32)
            return time, flux, flux_err, _FakeWCS(), aperture, quality

    monkeypatch.setattr(
        "bittr_tess_vetter.cli.localize_cli._resolve_candidate_inputs",
        _fake_resolve_candidate_inputs,
    )
    monkeypatch.setattr("bittr_tess_vetter.cli.localize_cli.MASTClient", _FakeMASTClient)
    monkeypatch.setattr("bittr_tess_vetter.cli.localize_cli._select_tpf_sectors", lambda **_kwargs: [14])
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.localize_cli.lookup_tic_coordinates",
        lambda tic_id: (_ for _ in ()).throw(AssertionError(f"lookup should not run for TIC {tic_id}")),
    )

    runner = CliRunner()
    result = runner.invoke(
        localize_command,
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
    assert result.exit_code == 1
    assert "Target coordinates unavailable" in result.output


def test_btv_localize_cli_coordinates_set_coordinate_source_cli(monkeypatch, tmp_path: Path) -> None:
    def _fake_resolve_candidate_inputs(**_kwargs: Any):
        return 123, 10.5, 2000.2, 2.5, None, {"source": "cli"}

    class _FakeWCS:
        def to_header(self, relax: bool = True):
            _ = relax
            return {}

    class _FakeMASTClient:
        def download_all_sectors(self, tic_id: int, flux_type: str, sectors: list[int] | None):
            _ = flux_type, sectors
            return [_make_lc_data(tic_id=tic_id, sector=14)]

        def download_tpf_cached(self, tic_id: int, sector: int):
            _ = tic_id, sector
            n = 16
            time = np.linspace(2000.0, 2001.0, n, dtype=np.float64)
            flux = np.ones((n, 3, 3), dtype=np.float64)
            flux_err = np.full((n, 3, 3), 1e-3, dtype=np.float64)
            aperture = np.ones((3, 3), dtype=np.int32)
            quality = np.zeros(n, dtype=np.int32)
            return time, flux, flux_err, _FakeWCS(), aperture, quality

    class _FakeLocalizationResult:
        def to_dict(self) -> dict[str, Any]:
            return {"verdict": "ON_TARGET"}

    monkeypatch.setattr(
        "bittr_tess_vetter.cli.localize_cli._resolve_candidate_inputs",
        _fake_resolve_candidate_inputs,
    )
    monkeypatch.setattr("bittr_tess_vetter.cli.localize_cli.MASTClient", _FakeMASTClient)
    monkeypatch.setattr("bittr_tess_vetter.cli.localize_cli._select_tpf_sectors", lambda **_kwargs: [14])
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.localize_cli.lookup_tic_coordinates",
        lambda tic_id: (_ for _ in ()).throw(AssertionError(f"lookup should not run for TIC {tic_id}")),
    )
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.localize_cli.localize_transit_source",
        lambda **_kwargs: _FakeLocalizationResult(),
    )

    out_path = tmp_path / "localize_cli_coords.json"
    runner = CliRunner()
    result = runner.invoke(
        localize_command,
        [
            "--tic-id",
            "123",
            "--period-days",
            "10.5",
            "--t0-btjd",
            "2000.2",
            "--duration-hours",
            "2.5",
            "--ra-deg",
            "120.0",
            "--dec-deg",
            "-20.0",
            "--out",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["provenance"]["coordinate_source"] == "cli"


def test_btv_localize_lookup_data_unavailable_exits_4(monkeypatch) -> None:
    def _fake_resolve_candidate_inputs(**_kwargs: Any):
        return 123, 10.5, 2000.2, 2.5, None, {"source": "cli"}

    class _FakeWCS:
        def to_header(self, relax: bool = True):
            _ = relax
            return {}

    class _FakeMASTClient:
        def download_all_sectors(self, tic_id: int, flux_type: str, sectors: list[int] | None):
            _ = flux_type, sectors
            return [_make_lc_data(tic_id=tic_id, sector=14)]

        def download_tpf_cached(self, tic_id: int, sector: int):
            _ = tic_id, sector
            n = 16
            time = np.linspace(2000.0, 2001.0, n, dtype=np.float64)
            flux = np.ones((n, 3, 3), dtype=np.float64)
            flux_err = np.full((n, 3, 3), 1e-3, dtype=np.float64)
            aperture = np.ones((3, 3), dtype=np.int32)
            quality = np.zeros(n, dtype=np.int32)
            return time, flux, flux_err, _FakeWCS(), aperture, quality

    class _LookupResult:
        status = LookupStatus.DATA_UNAVAILABLE
        ra_deg = None
        dec_deg = None
        message = "no coordinates"

    monkeypatch.setattr(
        "bittr_tess_vetter.cli.localize_cli._resolve_candidate_inputs",
        _fake_resolve_candidate_inputs,
    )
    monkeypatch.setattr("bittr_tess_vetter.cli.localize_cli.MASTClient", _FakeMASTClient)
    monkeypatch.setattr("bittr_tess_vetter.cli.localize_cli._select_tpf_sectors", lambda **_kwargs: [14])
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.localize_cli.lookup_tic_coordinates",
        lambda tic_id: _LookupResult(),
    )

    runner = CliRunner()
    result = runner.invoke(
        localize_command,
        [
            "--tic-id",
            "123",
            "--period-days",
            "10.5",
            "--t0-btjd",
            "2000.2",
            "--duration-hours",
            "2.5",
            "--network-ok",
        ],
    )
    assert result.exit_code == 4
    assert "no coordinates" in result.output
