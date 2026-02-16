from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from click.testing import CliRunner

import bittr_tess_vetter.cli.enrich_cli as enrich_cli
from bittr_tess_vetter.domain.lightcurve import LightCurveData
from bittr_tess_vetter.platform.io.mast_client import LightCurveNotFoundError


def _make_lc(*, tic_id: int, sector: int, start: float) -> LightCurveData:
    time = np.linspace(start, start + 1.0, 32, dtype=np.float64)
    flux = np.ones_like(time, dtype=np.float64)
    flux_err = np.full_like(time, 1e-3, dtype=np.float64)
    quality = np.zeros(time.shape, dtype=np.int32)
    valid_mask = np.ones(time.shape, dtype=np.bool_)
    return LightCurveData(
        time=time,
        flux=flux,
        flux_err=flux_err,
        quality=quality,
        valid_mask=valid_mask,
        tic_id=int(tic_id),
        sector=int(sector),
        cadence_seconds=120.0,
    )


def test_btv_help_lists_fpp() -> None:
    runner = CliRunner()
    result = runner.invoke(enrich_cli.cli, ["--help"])
    assert result.exit_code == 0
    assert "fpp" in result.output


def test_btv_fpp_success_plumbs_api_params_and_emits_contract(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def _fake_build_cache_for_fpp(**_kwargs: Any) -> tuple[object, list[int]]:
        return object(), [14, 15]

    def _fake_calculate_fpp(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {
            "fpp": 0.123,
            "nfpp": 0.001,
            "disposition": "possible_planet",
            "base_seed": 99,
        }

    monkeypatch.setattr(
        "bittr_tess_vetter.cli.fpp_cli._build_cache_for_fpp",
        _fake_build_cache_for_fpp,
    )
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.fpp_cli.calculate_fpp",
        _fake_calculate_fpp,
    )

    out_path = tmp_path / "fpp.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "fpp",
            "--tic-id",
            "123",
            "--period-days",
            "7.5",
            "--t0-btjd",
            "2500.25",
            "--duration-hours",
            "3.0",
            "--depth-ppm",
            "900.0",
            "--preset",
            "standard",
            "--replicates",
            "4",
            "--seed",
            "99",
            "--sectors",
            "14",
            "--sectors",
            "15",
            "--timeout-seconds",
            "120",
            "--network-ok",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["tic_id"] == 123
    assert seen["period"] == 7.5
    assert seen["t0"] == 2500.25
    assert seen["duration_hours"] == 3.0
    assert seen["depth_ppm"] == 900.0
    assert seen["preset"] == "standard"
    assert seen["replicates"] == 4
    assert seen["seed"] == 99
    assert seen["sectors"] == [14, 15]
    assert seen["timeout_seconds"] == 120.0
    assert seen["stellar_radius"] is None
    assert seen["stellar_mass"] is None
    assert seen["tmag"] is None

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "cli.fpp.v2"
    assert "fpp_result" in payload
    assert "provenance" in payload
    assert "inputs" in payload["provenance"]
    assert payload["provenance"]["depth_source"] == "explicit"
    assert payload["provenance"]["depth_ppm_used"] == 900.0
    assert payload["provenance"]["resolved_source"] == "cli"
    assert payload["provenance"]["runtime"]["preset"] == "standard"
    assert payload["provenance"]["runtime"]["seed_requested"] == 99
    assert payload["provenance"]["runtime"]["seed_effective"] == 99
    assert payload["provenance"]["runtime"]["timeout_seconds_requested"] == 120.0
    assert payload["provenance"]["runtime"]["timeout_seconds"] == 120.0


def test_btv_fpp_standard_preset_defaults_timeout_900(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def _fake_build_cache_for_fpp(**_kwargs: Any) -> tuple[object, list[int]]:
        return object(), [14, 15]

    def _fake_calculate_fpp(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {"fpp": 0.123, "nfpp": 0.001, "base_seed": 99}

    monkeypatch.setattr("bittr_tess_vetter.cli.fpp_cli._build_cache_for_fpp", _fake_build_cache_for_fpp)
    monkeypatch.setattr("bittr_tess_vetter.cli.fpp_cli.calculate_fpp", _fake_calculate_fpp)

    out_path = tmp_path / "fpp_standard_default_timeout.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "fpp",
            "--tic-id",
            "123",
            "--period-days",
            "7.5",
            "--t0-btjd",
            "2500.25",
            "--duration-hours",
            "3.0",
            "--depth-ppm",
            "900.0",
            "--preset",
            "standard",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["timeout_seconds"] == 900.0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["provenance"]["runtime"]["timeout_seconds_requested"] is None
    assert payload["provenance"]["runtime"]["timeout_seconds"] == 900.0


def test_btv_fpp_tutorial_preset_plumbs_and_runtime_reflects_tutorial(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def _fake_build_cache_for_fpp(**_kwargs: Any) -> tuple[object, list[int]]:
        return object(), [14, 15]

    def _fake_calculate_fpp(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {"fpp": 0.02, "nfpp": 0.001, "base_seed": 7}

    monkeypatch.setattr("bittr_tess_vetter.cli.fpp_cli._build_cache_for_fpp", _fake_build_cache_for_fpp)
    monkeypatch.setattr("bittr_tess_vetter.cli.fpp_cli.calculate_fpp", _fake_calculate_fpp)

    out_path = tmp_path / "fpp_tutorial_preset.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "fpp",
            "--tic-id",
            "123",
            "--period-days",
            "7.5",
            "--t0-btjd",
            "2500.25",
            "--duration-hours",
            "3.0",
            "--depth-ppm",
            "900.0",
            "--preset",
            "tutorial",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["preset"] == "tutorial"
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["provenance"]["runtime"]["preset"] == "tutorial"


def test_btv_fpp_standard_degenerate_emits_retry_guidance(monkeypatch, tmp_path: Path) -> None:
    def _fake_build_cache_for_fpp(**_kwargs: Any) -> tuple[object, list[int]]:
        return object(), [14, 15]

    def _fake_calculate_fpp(**_kwargs: Any) -> dict[str, Any]:
        return {
            "fpp": float("nan"),
            "nfpp": 0.1,
            "base_seed": 7,
            "degenerate_reason": "fpp_not_finite,posterior_sum_not_finite,posterior_prob_nan_count=30",
            "posterior_sum_total": float("nan"),
            "posterior_prob_nan_count": 30,
        }

    monkeypatch.setattr("bittr_tess_vetter.cli.fpp_cli._build_cache_for_fpp", _fake_build_cache_for_fpp)
    monkeypatch.setattr("bittr_tess_vetter.cli.fpp_cli.calculate_fpp", _fake_calculate_fpp)

    out_path = tmp_path / "fpp_standard_degenerate_retry_guidance.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "fpp",
            "--tic-id",
            "123",
            "--period-days",
            "7.5",
            "--t0-btjd",
            "2500.25",
            "--duration-hours",
            "3.0",
            "--depth-ppm",
            "900.0",
            "--preset",
            "standard",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    guidance = payload["provenance"]["retry_guidance"]
    assert guidance["preset"] == "tutorial"
    assert guidance["overrides"] == {
        "mc_draws": 200000,
        "max_points": 3000,
        "window_duration_mult": 2.0,
        "min_flux_err": 5e-5,
        "use_empirical_noise_floor": True,
    }
    assert guidance["reason"] == "fpp_not_finite,posterior_sum_not_finite,posterior_prob_nan_count=30"


def test_btv_fpp_standard_non_degenerate_omits_retry_guidance(monkeypatch, tmp_path: Path) -> None:
    def _fake_build_cache_for_fpp(**_kwargs: Any) -> tuple[object, list[int]]:
        return object(), [14, 15]

    def _fake_calculate_fpp(**_kwargs: Any) -> dict[str, Any]:
        return {
            "fpp": 0.01,
            "nfpp": 0.001,
            "base_seed": 7,
            "degenerate_reason": None,
            "posterior_sum_total": 1.0,
            "posterior_prob_nan_count": 0,
        }

    monkeypatch.setattr("bittr_tess_vetter.cli.fpp_cli._build_cache_for_fpp", _fake_build_cache_for_fpp)
    monkeypatch.setattr("bittr_tess_vetter.cli.fpp_cli.calculate_fpp", _fake_calculate_fpp)

    out_path = tmp_path / "fpp_standard_non_degenerate.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "fpp",
            "--tic-id",
            "123",
            "--period-days",
            "7.5",
            "--t0-btjd",
            "2500.25",
            "--duration-hours",
            "3.0",
            "--depth-ppm",
            "900.0",
            "--preset",
            "standard",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert "retry_guidance" not in payload["provenance"]


def test_btv_fpp_contrast_curve_tbl_parsed_and_passed(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def _fake_build_cache_for_fpp(**_kwargs: Any) -> tuple[object, list[int]]:
        return object(), [14]

    def _fake_calculate_fpp(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {"fpp": 0.12, "nfpp": 0.01, "base_seed": 7}

    monkeypatch.setattr("bittr_tess_vetter.cli.fpp_cli._build_cache_for_fpp", _fake_build_cache_for_fpp)
    monkeypatch.setattr("bittr_tess_vetter.cli.fpp_cli.calculate_fpp", _fake_calculate_fpp)

    cc_path = tmp_path / "contrast.tbl"
    cc_path.write_text(
        "\n".join(
            [
                "# ExoFOP contrast curve",
                "0.10 1.5",
                "0.50 4.2",
                "1.00 6.0",
            ]
        ),
        encoding="utf-8",
    )

    out_path = tmp_path / "fpp_with_contrast.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "fpp",
            "--tic-id",
            "123",
            "--period-days",
            "7.5",
            "--t0-btjd",
            "2500.25",
            "--duration-hours",
            "3.0",
            "--depth-ppm",
            "900.0",
            "--contrast-curve",
            str(cc_path),
            "--contrast-curve-filter",
            "Kcont",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    cc = seen["contrast_curve"]
    assert cc is not None
    assert cc.filter == "Kcont"
    assert len(cc.separation_arcsec) == 3
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["provenance"]["contrast_curve"]["path"] == str(cc_path)
    assert payload["provenance"]["contrast_curve"]["filter"] == "Kcont"


def test_btv_fpp_overrides_are_forwarded(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def _fake_build_cache_for_fpp(**_kwargs: Any) -> tuple[object, list[int]]:
        return object(), [14]

    def _fake_calculate_fpp(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {"fpp": 0.12, "nfpp": 0.01, "base_seed": 7}

    monkeypatch.setattr("bittr_tess_vetter.cli.fpp_cli._build_cache_for_fpp", _fake_build_cache_for_fpp)
    monkeypatch.setattr("bittr_tess_vetter.cli.fpp_cli.calculate_fpp", _fake_calculate_fpp)

    out_path = tmp_path / "fpp_with_overrides.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "fpp",
            "--tic-id",
            "123",
            "--period-days",
            "7.5",
            "--t0-btjd",
            "2500.25",
            "--duration-hours",
            "3.0",
            "--depth-ppm",
            "900.0",
            "--override",
            "mc_draws=200000",
            "--override",
            "use_empirical_noise_floor=true",
            "--out",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["overrides"] == {"mc_draws": 200000, "use_empirical_noise_floor": True}
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["provenance"]["runtime"]["overrides"] == {
        "mc_draws": 200000,
        "use_empirical_noise_floor": True,
    }


def test_btv_fpp_detrend_cache_requires_detrend(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "fpp",
            "--tic-id",
            "123",
            "--period-days",
            "7.5",
            "--t0-btjd",
            "2500.25",
            "--duration-hours",
            "3.0",
            "--depth-ppm",
            "900.0",
            "--detrend-cache",
            "--out",
            str(tmp_path / "fpp.json"),
        ],
    )
    assert result.exit_code == 1
    assert "--detrend-cache requires --detrend" in result.output


def test_btv_fpp_timeout_maps_to_exit_5(monkeypatch) -> None:
    def _fake_build_cache_for_fpp(**_kwargs: Any) -> tuple[object, list[int]]:
        return object(), [1]

    def _timeout(**_kwargs: Any) -> dict[str, Any]:
        raise TimeoutError("timed out")

    monkeypatch.setattr(
        "bittr_tess_vetter.cli.fpp_cli._build_cache_for_fpp",
        _fake_build_cache_for_fpp,
    )
    monkeypatch.setattr("bittr_tess_vetter.cli.fpp_cli.calculate_fpp", _timeout)

    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "fpp",
            "--tic-id",
            "123",
            "--period-days",
            "7.5",
            "--t0-btjd",
            "2500.25",
            "--duration-hours",
            "3.0",
            "--depth-ppm",
            "900.0",
        ],
    )

    assert result.exit_code == 5


def test_btv_fpp_missing_depth_from_toi_maps_to_exit_4(monkeypatch) -> None:
    def _fake_resolve_candidate_inputs(**_kwargs: Any):
        return 123, 7.5, 2500.25, 3.0, None, {"source": "toi_catalog", "resolved_from": "exofop"}

    monkeypatch.setattr(
        "bittr_tess_vetter.cli.fpp_cli._resolve_candidate_inputs",
        _fake_resolve_candidate_inputs,
    )

    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "fpp",
            "--toi",
            "TOI-123.01",
            "--network-ok",
        ],
    )

    assert result.exit_code == 4


def test_btv_fpp_uses_detrended_depth_before_catalog(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def _fake_resolve_candidate_inputs(**_kwargs: Any):
        return 123, 7.5, 2500.25, 3.0, 600.0, {"source": "toi_catalog", "resolved_from": "exofop"}

    def _fake_build_cache_for_fpp(**_kwargs: Any) -> tuple[object, list[int]]:
        return object(), [14]

    def _fake_calculate_fpp(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {"fpp": 0.12, "nfpp": 0.01, "base_seed": 7}

    def _fake_detrended_depth(**_kwargs: Any) -> tuple[float | None, dict[str, Any]]:
        return 777.0, {"method": "transit_masked_bin_median"}

    monkeypatch.setattr("bittr_tess_vetter.cli.fpp_cli._resolve_candidate_inputs", _fake_resolve_candidate_inputs)
    monkeypatch.setattr("bittr_tess_vetter.cli.fpp_cli._build_cache_for_fpp", _fake_build_cache_for_fpp)
    monkeypatch.setattr("bittr_tess_vetter.cli.fpp_cli.calculate_fpp", _fake_calculate_fpp)
    monkeypatch.setattr("bittr_tess_vetter.cli.fpp_cli._estimate_detrended_depth_ppm", _fake_detrended_depth)

    out_path = tmp_path / "detrended.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "fpp",
            "--tic-id",
            "123",
            "--period-days",
            "7.5",
            "--t0-btjd",
            "2500.25",
            "--duration-hours",
            "3.0",
            "--detrend",
            "transit_masked_bin_median",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["depth_ppm"] == 777.0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["provenance"]["depth_source"] == "detrended"
    assert payload["provenance"]["depth_ppm_used"] == 777.0
    assert payload["provenance"]["inputs"]["depth_ppm_catalog"] == 600.0


def test_btv_fpp_depth_precedence_explicit_over_detrended(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {"detrended_called": False}

    def _fake_resolve_candidate_inputs(**_kwargs: Any):
        return 123, 7.5, 2500.25, 3.0, 900.0, {"source": "cli", "resolved_from": "cli"}

    def _fake_build_cache_for_fpp(**_kwargs: Any) -> tuple[object, list[int]]:
        return object(), [14]

    def _fake_calculate_fpp(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {"fpp": 0.12, "nfpp": 0.01, "base_seed": 7}

    def _fake_detrended_depth(**_kwargs: Any) -> tuple[float | None, dict[str, Any]]:
        seen["detrended_called"] = True
        return 777.0, {"method": "transit_masked_bin_median"}

    monkeypatch.setattr("bittr_tess_vetter.cli.fpp_cli._resolve_candidate_inputs", _fake_resolve_candidate_inputs)
    monkeypatch.setattr("bittr_tess_vetter.cli.fpp_cli._build_cache_for_fpp", _fake_build_cache_for_fpp)
    monkeypatch.setattr("bittr_tess_vetter.cli.fpp_cli.calculate_fpp", _fake_calculate_fpp)
    monkeypatch.setattr("bittr_tess_vetter.cli.fpp_cli._estimate_detrended_depth_ppm", _fake_detrended_depth)

    out_path = tmp_path / "explicit.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "fpp",
            "--tic-id",
            "123",
            "--period-days",
            "7.5",
            "--t0-btjd",
            "2500.25",
            "--duration-hours",
            "3.0",
            "--depth-ppm",
            "900.0",
            "--detrend",
            "transit_masked_bin_median",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["depth_ppm"] == 900.0
    assert seen["detrended_called"] is False
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["provenance"]["depth_source"] == "explicit"


def test_btv_fpp_stellar_precedence_explicit_over_file_over_auto(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}
    stellar_file = tmp_path / "stellar.json"
    stellar_file.write_text(
        json.dumps({"stellar": {"radius": 0.8, "mass": 0.7, "tmag": 10.2}}),
        encoding="utf-8",
    )

    def _fake_build_cache_for_fpp(**_kwargs: Any) -> tuple[object, list[int]]:
        return object(), [1]

    def _fake_calculate_fpp(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {"fpp": 0.2, "nfpp": 0.01, "base_seed": 5}

    def _fake_auto(_tic_id: int) -> dict[str, float | None]:
        return {"radius": 1.1, "mass": 1.0, "tmag": 11.0}

    monkeypatch.setattr("bittr_tess_vetter.cli.fpp_cli._build_cache_for_fpp", _fake_build_cache_for_fpp)
    monkeypatch.setattr("bittr_tess_vetter.cli.fpp_cli.calculate_fpp", _fake_calculate_fpp)
    monkeypatch.setattr("bittr_tess_vetter.cli.fpp_cli._load_auto_stellar_inputs", _fake_auto)

    out_path = tmp_path / "stellar_out.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "fpp",
            "--tic-id",
            "123",
            "--period-days",
            "7.5",
            "--t0-btjd",
            "2500.25",
            "--duration-hours",
            "3.0",
            "--depth-ppm",
            "900.0",
            "--stellar-file",
            str(stellar_file),
            "--stellar-mass",
            "0.95",
            "--use-stellar-auto",
            "--network-ok",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["stellar_radius"] == 0.8
    assert seen["stellar_mass"] == 0.95
    assert seen["tmag"] == 10.2
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["provenance"]["stellar"]["sources"] == {
        "radius": "file",
        "mass": "explicit",
        "tmag": "file",
    }


def test_btv_fpp_lightcurve_missing_maps_to_exit_4(monkeypatch) -> None:
    def _missing_cache(**_kwargs: Any):
        raise LightCurveNotFoundError("missing sectors")

    monkeypatch.setattr(
        "bittr_tess_vetter.cli.fpp_cli._build_cache_for_fpp",
        _missing_cache,
    )

    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "fpp",
            "--tic-id",
            "123",
            "--period-days",
            "7.5",
            "--t0-btjd",
            "2500.25",
            "--duration-hours",
            "3.0",
            "--depth-ppm",
            "900.0",
        ],
    )

    assert result.exit_code == 4


def test_build_cache_for_fpp_stores_requested_sector_products(monkeypatch, tmp_path: Path) -> None:
    class _FakeLC:
        def __init__(self, sector: int) -> None:
            self.sector = sector

    class _FakeMASTClient:
        def download_all_sectors(self, tic_id: int, *, flux_type: str, sectors: list[int] | None = None):
            assert tic_id == 123
            assert flux_type == "pdcsap"
            assert sectors == [14, 15]
            return [_FakeLC(14), _FakeLC(15)]

    class _FakePersistentCache:
        def __init__(self, cache_dir: Path | None = None) -> None:
            self.cache_dir = cache_dir
            self.records: dict[str, object] = {}

        def put(self, key: str, value: object) -> None:
            self.records[key] = value

    monkeypatch.setattr("bittr_tess_vetter.cli.fpp_cli.MASTClient", _FakeMASTClient)
    monkeypatch.setattr("bittr_tess_vetter.cli.fpp_cli.PersistentCache", _FakePersistentCache)

    from bittr_tess_vetter.cli.fpp_cli import _build_cache_for_fpp

    cache, loaded = _build_cache_for_fpp(tic_id=123, sectors=[14, 15], cache_dir=tmp_path)
    assert loaded == [14, 15]
    assert isinstance(cache, _FakePersistentCache)
    assert sorted(cache.records.keys()) == [
        "lc:123:14:pdcsap",
        "lc:123:15:pdcsap",
    ]


def test_btv_fpp_report_file_precedence_and_sector_fallback(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": "cli.vet.v2",
                "inputs_summary": {
                    "input_resolution": {
                        "inputs": {
                            "tic_id": 222,
                            "period_days": 2.5,
                            "t0_btjd": 1300.25,
                            "duration_hours": 1.5,
                            "depth_ppm": 600.0,
                        }
                    }
                },
                "provenance": {"sectors_used": [20, 21]},
            }
        ),
        encoding="utf-8",
    )

    def _fake_resolve_candidate_inputs(**kwargs: Any):
        seen["resolve"] = kwargs
        return 333, 2.5, 1300.25, 1.5, 900.0, {"source": "cli", "resolved_from": "cli"}

    def _fake_build_cache_for_fpp(**kwargs: Any) -> tuple[object, list[int]]:
        seen["build_cache"] = kwargs
        return object(), [20, 21]

    def _fake_calculate_fpp(**kwargs: Any) -> dict[str, Any]:
        seen["fpp"] = kwargs
        return {"fpp": 0.12, "nfpp": 0.01, "base_seed": 7}

    monkeypatch.setattr("bittr_tess_vetter.cli.fpp_cli._resolve_candidate_inputs", _fake_resolve_candidate_inputs)
    monkeypatch.setattr("bittr_tess_vetter.cli.fpp_cli._build_cache_for_fpp", _fake_build_cache_for_fpp)
    monkeypatch.setattr("bittr_tess_vetter.cli.fpp_cli.calculate_fpp", _fake_calculate_fpp)

    out_path = tmp_path / "fpp_report_precedence.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "fpp",
            "--report-file",
            str(report_path),
            "--tic-id",
            "333",
            "--depth-ppm",
            "900.0",
            "--toi",
            "TOI-123.01",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "--report-file provided with --toi" in result.output
    assert seen["resolve"]["toi"] is None
    assert seen["resolve"]["tic_id"] == 333
    assert seen["resolve"]["period_days"] == 2.5
    assert seen["resolve"]["t0_btjd"] == 1300.25
    assert seen["resolve"]["duration_hours"] == 1.5
    assert seen["resolve"]["depth_ppm"] == 900.0
    assert seen["build_cache"]["sectors"] == [20, 21]
    assert seen["fpp"]["sectors"] == [20, 21]

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["provenance"]["inputs"]["sectors"] == [20, 21]


def test_btv_fpp_explicit_sectors_use_network_download_by_default(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {"cached_calls": 0, "network_calls": 0}

    class _FakeMASTClient:
        def download_all_sectors(self, *_args: Any, **_kwargs: Any):
            seen["network_calls"] += 1
            return [
                _make_lc(tic_id=123, sector=14, start=2014.0),
                _make_lc(tic_id=123, sector=15, start=2015.0),
            ]

        def download_lightcurve_cached(self, *_args: Any, **_kwargs: Any):
            seen["cached_calls"] += 1
            raise AssertionError("cached loader should not be used by default")

    def _fake_detrend_lightcurve_for_vetting(**kwargs: Any):
        lc = kwargs["lc"]
        return lc, {"method": kwargs["method"]}

    def _fake_measure_transit_depth(*_args: Any, **_kwargs: Any):
        return 8e-4, 1e-4

    def _fake_calculate_fpp(**_kwargs: Any) -> dict[str, Any]:
        return {"fpp": 0.12, "nfpp": 0.01, "base_seed": 7}

    monkeypatch.setattr("bittr_tess_vetter.cli.fpp_cli.MASTClient", _FakeMASTClient)
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.fpp_cli._detrend_lightcurve_for_vetting",
        _fake_detrend_lightcurve_for_vetting,
    )
    monkeypatch.setattr("bittr_tess_vetter.cli.fpp_cli.measure_transit_depth", _fake_measure_transit_depth)
    monkeypatch.setattr("bittr_tess_vetter.cli.fpp_cli.calculate_fpp", _fake_calculate_fpp)

    out_path = tmp_path / "fpp_explicit_sectors_download_default.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "fpp",
            "--tic-id",
            "123",
            "--period-days",
            "7.5",
            "--t0-btjd",
            "2500.25",
            "--duration-hours",
            "3.0",
            "--detrend",
            "transit_masked_bin_median",
            "--sectors",
            "14",
            "--sectors",
            "15",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["network_calls"] >= 2
    assert seen["cached_calls"] == 0


def test_btv_fpp_explicit_sectors_cache_only_when_requested(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {"cached_calls": 0, "network_calls": 0}

    class _FakeMASTClient:
        def download_all_sectors(self, *_args: Any, **_kwargs: Any):
            seen["network_calls"] += 1
            raise AssertionError("download_all_sectors should not be used when --sectors is explicit")

        def download_lightcurve_cached(self, tic_id: int, sector: int, flux_type: str):
            assert flux_type == "pdcsap"
            seen["cached_calls"] += 1
            return _make_lc(tic_id=tic_id, sector=sector, start=2000.0 + float(sector))

    def _fake_detrend_lightcurve_for_vetting(**kwargs: Any):
        lc = kwargs["lc"]
        return lc, {"method": kwargs["method"]}

    def _fake_measure_transit_depth(*_args: Any, **_kwargs: Any):
        return 8e-4, 1e-4

    def _fake_calculate_fpp(**_kwargs: Any) -> dict[str, Any]:
        return {"fpp": 0.12, "nfpp": 0.01, "base_seed": 7}

    monkeypatch.setattr("bittr_tess_vetter.cli.fpp_cli.MASTClient", _FakeMASTClient)
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.fpp_cli._detrend_lightcurve_for_vetting",
        _fake_detrend_lightcurve_for_vetting,
    )
    monkeypatch.setattr("bittr_tess_vetter.cli.fpp_cli.measure_transit_depth", _fake_measure_transit_depth)
    monkeypatch.setattr("bittr_tess_vetter.cli.fpp_cli.calculate_fpp", _fake_calculate_fpp)

    out_path = tmp_path / "fpp_explicit_sectors_cache_only.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "fpp",
            "--tic-id",
            "123",
            "--period-days",
            "7.5",
            "--t0-btjd",
            "2500.25",
            "--duration-hours",
            "3.0",
            "--detrend",
            "transit_masked_bin_median",
            "--sectors",
            "14",
            "--sectors",
            "15",
            "--cache-only-sectors",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["network_calls"] == 0
    assert seen["cached_calls"] >= 4


def test_btv_fpp_supports_positional_toi_and_short_o(monkeypatch, tmp_path: Path) -> None:
    def _fake_build_cache_for_fpp(**_kwargs: Any) -> tuple[object, list[int]]:
        return object(), [14, 15]

    def _fake_calculate_fpp(**_kwargs: Any) -> dict[str, Any]:
        return {"fpp": 0.12, "nfpp": 0.01, "base_seed": 7}

    monkeypatch.setattr("bittr_tess_vetter.cli.fpp_cli._build_cache_for_fpp", _fake_build_cache_for_fpp)
    monkeypatch.setattr("bittr_tess_vetter.cli.fpp_cli.calculate_fpp", _fake_calculate_fpp)

    out_path = tmp_path / "fpp_positional_toi_short_o.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "fpp",
            "TOI-123.01",
            "--tic-id",
            "123",
            "--period-days",
            "7.5",
            "--t0-btjd",
            "2500.25",
            "--duration-hours",
            "3.0",
            "--depth-ppm",
            "900.0",
            "-o",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "cli.fpp.v2"
