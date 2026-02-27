from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from click.testing import CliRunner

import tess_vetter.cli.enrich_cli as enrich_cli
from tess_vetter.domain.lightcurve import LightCurveData, make_data_ref
from tess_vetter.platform.io.mast_client import LightCurveNotFoundError


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


def test_build_cache_for_fpp_prefers_persistent_cache_for_requested_sectors(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from tess_vetter.cli import fpp_cli
    from tess_vetter.platform.io import PersistentCache

    seen = {"network_calls": 0, "cached_calls": 0}

    class _FakeMASTClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
            pass

        def download_all_sectors(self, *_args: Any, **_kwargs: Any):
            seen["network_calls"] += 1
            raise AssertionError("network should not be used when persistent cache has requested sectors")

        def download_lightcurve_cached(self, *_args: Any, **_kwargs: Any):
            seen["cached_calls"] += 1
            raise AssertionError("lightkurve cache should not be needed when persistent cache is warm")

    cache = PersistentCache(cache_dir=tmp_path)
    cache.put(make_data_ref(123, 14, "pdcsap"), _make_lc(tic_id=123, sector=14, start=2000.0))
    cache.put(make_data_ref(123, 15, "pdcsap"), _make_lc(tic_id=123, sector=15, start=2001.0))

    monkeypatch.setattr("tess_vetter.cli.fpp_cli.MASTClient", _FakeMASTClient)

    built_cache, sectors_loaded = fpp_cli._build_cache_for_fpp(
        tic_id=123,
        sectors=[14, 15],
        cache_dir=tmp_path,
    )

    assert sectors_loaded == [14, 15]
    assert seen["network_calls"] == 0
    assert seen["cached_calls"] == 0
    assert built_cache.get(make_data_ref(123, 14, "pdcsap")) is not None


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
        "tess_vetter.cli.fpp_cli._build_cache_for_fpp",
        _fake_build_cache_for_fpp,
    )
    monkeypatch.setattr(
        "tess_vetter.cli.fpp_cli.calculate_fpp",
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
    assert seen["allow_network"] is True
    assert seen["stellar_radius"] is None
    assert seen["stellar_mass"] is None
    assert seen["tmag"] is None

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "cli.fpp.v3"
    assert "fpp_result" in payload
    assert payload["verdict"] == "FPP_POSSIBLE_PLANET"
    assert payload["verdict_source"] == "$.fpp_result.disposition"
    assert payload["result"]["verdict"] == payload["verdict"]
    assert payload["result"]["verdict_source"] == payload["verdict_source"]
    assert payload["result"]["fpp_result"] == payload["fpp_result"]
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

    monkeypatch.setattr("tess_vetter.cli.fpp_cli._build_cache_for_fpp", _fake_build_cache_for_fpp)
    monkeypatch.setattr("tess_vetter.cli.fpp_cli.calculate_fpp", _fake_calculate_fpp)

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
    assert payload["verdict"] == "FPP_HIGH"
    assert payload["verdict_source"] == "$.fpp_result.fpp"
    assert payload["result"]["verdict"] == payload["verdict"]
    assert payload["result"]["verdict_source"] == payload["verdict_source"]
    assert payload["provenance"]["runtime"]["timeout_seconds_requested"] is None
    assert payload["provenance"]["runtime"]["timeout_seconds"] == 900.0


def test_btv_fpp_tutorial_preset_plumbs_and_runtime_reflects_tutorial(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def _fake_build_cache_for_fpp(**_kwargs: Any) -> tuple[object, list[int]]:
        return object(), [14, 15]

    def _fake_calculate_fpp(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {"fpp": 0.02, "nfpp": 0.001, "base_seed": 7}

    monkeypatch.setattr("tess_vetter.cli.fpp_cli._build_cache_for_fpp", _fake_build_cache_for_fpp)
    monkeypatch.setattr("tess_vetter.cli.fpp_cli.calculate_fpp", _fake_calculate_fpp)

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

    monkeypatch.setattr("tess_vetter.cli.fpp_cli._build_cache_for_fpp", _fake_build_cache_for_fpp)
    monkeypatch.setattr("tess_vetter.cli.fpp_cli.calculate_fpp", _fake_calculate_fpp)

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
    assert guidance["overrides"]["mc_draws"] == 200000
    assert guidance["overrides"]["target_points"] == 3000
    assert guidance["overrides"]["window_duration_mult"] == 2.0
    assert guidance["overrides"]["min_flux_err"] == 5e-5
    assert guidance["overrides"]["use_empirical_noise_floor"] is True
    assert guidance["overrides"]["point_reduction"] == "downsample"
    assert guidance["overrides"]["bin_stat"] == "mean"
    assert guidance["overrides"]["bin_err"] == "propagate"
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

    monkeypatch.setattr("tess_vetter.cli.fpp_cli._build_cache_for_fpp", _fake_build_cache_for_fpp)
    monkeypatch.setattr("tess_vetter.cli.fpp_cli.calculate_fpp", _fake_calculate_fpp)

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


def test_btv_fpp_degenerate_guard_succeeds_on_retry_with_reduced_target_points(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("BTV_FPP_DEGENERATE_FALLBACK", "1")
    seen_overrides: list[dict[str, Any]] = []

    def _fake_build_cache_for_fpp(**_kwargs: Any) -> tuple[object, list[int]]:
        return object(), [14, 15]

    def _fake_calculate_fpp(**kwargs: Any) -> dict[str, Any]:
        seen_overrides.append(dict(kwargs["overrides"]))
        if len(seen_overrides) == 1:
            return {
                "fpp": float("nan"),
                "nfpp": 0.1,
                "base_seed": 7,
                "degenerate_reason": "fpp_not_finite,posterior_prob_nan_count=12",
                "posterior_sum_total": 1.0,
                "posterior_prob_nan_count": 12,
            }
        return {
            "fpp": 0.0123,
            "nfpp": 0.0012,
            "base_seed": 7,
            "degenerate_reason": None,
            "posterior_sum_total": 1.0,
            "posterior_prob_nan_count": 0,
        }

    monkeypatch.setattr("tess_vetter.cli.fpp_cli._build_cache_for_fpp", _fake_build_cache_for_fpp)
    monkeypatch.setattr("tess_vetter.cli.fpp_cli.calculate_fpp", _fake_calculate_fpp)

    out_path = tmp_path / "fpp_guard_success_on_retry.json"
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
            "--override",
            "target_points=8000",
            "--override",
            "max_points=8000",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen_overrides[0]["target_points"] == 8000
    assert seen_overrides[0]["max_points"] == 8000
    assert seen_overrides[1]["max_points"] == 3000

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    runtime = payload["provenance"]["runtime"]
    assert runtime["overrides"]["target_points"] == 8000
    assert runtime["overrides"]["max_points"] == 8000
    guard = runtime["degenerate_guard"]
    assert guard["guard_triggered"] is True
    assert guard["explicit_max_points_override"] is True
    assert guard["initial_max_points"] == 8000
    assert guard["final_selected_attempt"] == 2
    assert guard["fallback_succeeded"] is True
    assert guard["attempts"][0]["max_points"] == 8000
    assert guard["attempts"][1]["max_points"] == 3000
    assert guard["attempts"][0]["target_points"] == 8000
    assert guard["attempts"][0]["degenerate"] is True
    assert guard["attempts"][1]["degenerate"] is False
    assert "retry_guidance" not in payload["provenance"]


def test_btv_fpp_degenerate_guard_failure_after_bounded_retries(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BTV_FPP_DEGENERATE_FALLBACK", "1")
    seen_overrides: list[dict[str, Any]] = []

    def _fake_build_cache_for_fpp(**_kwargs: Any) -> tuple[object, list[int]]:
        return object(), [14, 15]

    def _fake_calculate_fpp(**kwargs: Any) -> dict[str, Any]:
        current_overrides = dict(kwargs["overrides"])
        seen_overrides.append(current_overrides)
        target_points = current_overrides.get("target_points")
        return {
            "fpp": float("nan"),
            "nfpp": 0.1,
            "base_seed": 7,
            "degenerate_reason": (
                f"fpp_not_finite,posterior_prob_nan_count=8,target_points={target_points}"
            ),
            "posterior_sum_total": float("nan"),
            "posterior_prob_nan_count": 8,
        }

    monkeypatch.setattr("tess_vetter.cli.fpp_cli._build_cache_for_fpp", _fake_build_cache_for_fpp)
    monkeypatch.setattr("tess_vetter.cli.fpp_cli.calculate_fpp", _fake_calculate_fpp)

    out_path = tmp_path / "fpp_guard_failure_after_retries.json"
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
            "--override",
            "max_points=4000",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert [entry.get("max_points") for entry in seen_overrides] == [4000, 3000, 2000, 1500]

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    guard = payload["provenance"]["runtime"]["degenerate_guard"]
    assert guard["guard_triggered"] is True
    assert guard["fallback_succeeded"] is False
    assert guard["final_selected_attempt"] == 4
    assert [attempt["max_points"] for attempt in guard["attempts"]] == [4000, 3000, 2000, 1500]
    assert payload["provenance"]["retry_guidance"]["preset"] == "tutorial"


def test_btv_fpp_target_points_and_max_points_equal_accepts_and_warns(
    monkeypatch, tmp_path: Path
) -> None:
    seen: dict[str, Any] = {}

    def _fake_build_cache_for_fpp(**_kwargs: Any) -> tuple[object, list[int]]:
        return object(), [14]

    def _fake_calculate_fpp(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {"fpp": 0.02, "nfpp": 0.002, "base_seed": 11}

    monkeypatch.setattr("tess_vetter.cli.fpp_cli._build_cache_for_fpp", _fake_build_cache_for_fpp)
    monkeypatch.setattr("tess_vetter.cli.fpp_cli.calculate_fpp", _fake_calculate_fpp)

    out_path = tmp_path / "fpp_target_max_equal.json"
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
            "--target-points",
            "400",
            "--max-points",
            "400",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["overrides"]["target_points"] == 400
    assert "legacy alias" in result.output.lower()
    assert "--max-points" in result.output


def test_btv_fpp_target_points_and_max_points_mismatch_fails(tmp_path: Path) -> None:
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
            "--target-points",
            "400",
            "--max-points",
            "401",
            "--out",
            str(tmp_path / "fpp_target_max_mismatch.json"),
        ],
    )

    assert result.exit_code == 1
    assert "--target-points" in result.output
    assert "--max-points" in result.output
    assert "source of truth" in result.output.lower()


def test_btv_fpp_none_mode_warns_and_ignores_target_points(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def _fake_build_cache_for_fpp(**_kwargs: Any) -> tuple[object, list[int]]:
        return object(), [14]

    def _fake_calculate_fpp(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {"fpp": 0.03, "nfpp": 0.003, "base_seed": 12}

    monkeypatch.setattr("tess_vetter.cli.fpp_cli._build_cache_for_fpp", _fake_build_cache_for_fpp)
    monkeypatch.setattr("tess_vetter.cli.fpp_cli.calculate_fpp", _fake_calculate_fpp)

    out_path = tmp_path / "fpp_none_ignore_target_points.json"
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
            "--point-reduction",
            "none",
            "--target-points",
            "400",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["overrides"]["point_reduction"] == "none"
    assert seen["overrides"]["target_points"] == 400
    assert "ignores" in result.output.lower()
    assert "--target-points" in result.output


def test_btv_fpp_none_mode_warns_and_ignores_max_points_alias(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def _fake_build_cache_for_fpp(**_kwargs: Any) -> tuple[object, list[int]]:
        return object(), [14]

    def _fake_calculate_fpp(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {"fpp": 0.03, "nfpp": 0.003, "base_seed": 12}

    monkeypatch.setattr("tess_vetter.cli.fpp_cli._build_cache_for_fpp", _fake_build_cache_for_fpp)
    monkeypatch.setattr("tess_vetter.cli.fpp_cli.calculate_fpp", _fake_calculate_fpp)

    out_path = tmp_path / "fpp_none_ignore_max_points.json"
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
            "--point-reduction",
            "none",
            "--max-points",
            "400",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["overrides"]["point_reduction"] == "none"
    assert seen["overrides"]["target_points"] == 400
    assert "ignores" in result.output.lower()
    assert "--max-points" in result.output


def test_btv_fpp_contrast_curve_tbl_parsed_and_passed(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def _fake_build_cache_for_fpp(**_kwargs: Any) -> tuple[object, list[int]]:
        return object(), [14]

    def _fake_calculate_fpp(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {"fpp": 0.12, "nfpp": 0.01, "base_seed": 7}

    monkeypatch.setattr("tess_vetter.cli.fpp_cli._build_cache_for_fpp", _fake_build_cache_for_fpp)
    monkeypatch.setattr("tess_vetter.cli.fpp_cli.calculate_fpp", _fake_calculate_fpp)

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
    assert payload["provenance"]["contrast_curve"]["parse_provenance"]["strategy"] == "text_table_primary"


def test_btv_fpp_contrast_curve_fits_parsed_and_passed(monkeypatch, tmp_path: Path) -> None:
    fits = pytest.importorskip("astropy.io.fits")
    np = pytest.importorskip("numpy")
    seen: dict[str, Any] = {}

    def _fake_build_cache_for_fpp(**_kwargs: Any) -> tuple[object, list[int]]:
        return object(), [14]

    def _fake_calculate_fpp(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {"fpp": 0.12, "nfpp": 0.01, "base_seed": 7}

    monkeypatch.setattr("tess_vetter.cli.fpp_cli._build_cache_for_fpp", _fake_build_cache_for_fpp)
    monkeypatch.setattr("tess_vetter.cli.fpp_cli.calculate_fpp", _fake_calculate_fpp)

    cc_path = tmp_path / "TIC149302744I-at20190714_soarspeckle.fits"
    rng = np.random.default_rng(123)
    image = rng.normal(0.0, 0.001, size=(200, 200))
    image[100, 100] += 2.0
    fits.PrimaryHDU(data=image).writeto(cc_path)

    out_path = tmp_path / "fpp_with_contrast_fits.json"
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
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    cc = seen["contrast_curve"]
    assert cc is not None
    assert len(cc.separation_arcsec) >= 2
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["provenance"]["contrast_curve"]["path"] == str(cc_path)
    assert payload["provenance"]["contrast_curve"]["parse_provenance"]["strategy"] == "fits_image_azimuthal"


def test_btv_fpp_overrides_are_forwarded(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def _fake_build_cache_for_fpp(**_kwargs: Any) -> tuple[object, list[int]]:
        return object(), [14]

    def _fake_calculate_fpp(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {"fpp": 0.12, "nfpp": 0.01, "base_seed": 7}

    monkeypatch.setattr("tess_vetter.cli.fpp_cli._build_cache_for_fpp", _fake_build_cache_for_fpp)
    monkeypatch.setattr("tess_vetter.cli.fpp_cli.calculate_fpp", _fake_calculate_fpp)

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
    assert seen["overrides"]["mc_draws"] == 200000
    assert seen["overrides"]["use_empirical_noise_floor"] is True
    assert seen["overrides"]["point_reduction"] == "downsample"
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["provenance"]["runtime"]["overrides"]["mc_draws"] == 200000
    assert payload["provenance"]["runtime"]["overrides"]["use_empirical_noise_floor"] is True


def test_btv_fpp_overrides_bin_settings_and_target_points_trace(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def _fake_build_cache_for_fpp(**_kwargs: Any) -> tuple[object, list[int]]:
        return object(), [14]

    def _fake_calculate_fpp(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {"fpp": 0.12, "nfpp": 0.01, "base_seed": 7}

    monkeypatch.setattr("tess_vetter.cli.fpp_cli._build_cache_for_fpp", _fake_build_cache_for_fpp)
    monkeypatch.setattr("tess_vetter.cli.fpp_cli.calculate_fpp", _fake_calculate_fpp)

    out_path = tmp_path / "fpp_with_bin_overrides.json"
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
            "point_reduction=bin",
            "--override",
            "target_points=250",
            "--override",
            "bin_stat=median",
            "--override",
            "bin_err=robust",
            "--out",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["overrides"]["point_reduction"] == "bin"
    assert seen["overrides"]["target_points"] == 250
    assert seen["overrides"]["bin_stat"] == "median"
    assert seen["overrides"]["bin_err"] == "robust"
    assert seen["overrides"]["resolution_trace"]["target_points"]["source"] == "target_points"

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    runtime_overrides = payload["provenance"]["runtime"]["overrides"]
    assert runtime_overrides["bin_stat"] == "median"
    assert runtime_overrides["bin_err"] == "robust"
    assert runtime_overrides["resolution_trace"]["target_points"]["source"] == "target_points"


def test_btv_fpp_drop_scenario_forwarded(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def _fake_build_cache_for_fpp(**_kwargs: Any) -> tuple[object, list[int]]:
        return object(), [14]

    def _fake_calculate_fpp(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {"fpp": 0.12, "nfpp": 0.01, "base_seed": 7}

    monkeypatch.setattr("tess_vetter.cli.fpp_cli._build_cache_for_fpp", _fake_build_cache_for_fpp)
    monkeypatch.setattr("tess_vetter.cli.fpp_cli.calculate_fpp", _fake_calculate_fpp)

    out_path = tmp_path / "fpp_with_drop_scenario.json"
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
            "--drop-scenario",
            "EB",
            "--drop-scenario",
            "DEB",
            "--out",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["overrides"]["drop_scenario"] == ["EB", "DEB"]
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["provenance"]["runtime"]["overrides"]["drop_scenario"] == ["EB", "DEB"]


def test_btv_fpp_drop_scenario_mixed_case_label_accepted(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def _fake_build_cache_for_fpp(**_kwargs: Any) -> tuple[object, list[int]]:
        return object(), [14]

    def _fake_calculate_fpp(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {"fpp": 0.12, "nfpp": 0.01, "base_seed": 7}

    monkeypatch.setattr("tess_vetter.cli.fpp_cli._build_cache_for_fpp", _fake_build_cache_for_fpp)
    monkeypatch.setattr("tess_vetter.cli.fpp_cli.calculate_fpp", _fake_calculate_fpp)

    out_path = tmp_path / "fpp_with_drop_scenario_mixed_case.json"
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
            "--drop-scenario",
            "ebX2p",
            "--out",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["overrides"]["drop_scenario"] == ["EBx2P"]
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["provenance"]["runtime"]["overrides"]["drop_scenario"] == ["EBx2P"]


def test_btv_fpp_drop_scenario_invalid_label_rejected(tmp_path: Path) -> None:
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
            "--drop-scenario",
            "NOPE",
            "--out",
            str(tmp_path / "fpp_invalid_drop_scenario.json"),
        ],
    )
    assert result.exit_code == 1
    assert "Unknown drop_scenario label(s): NOPE." in result.output
    assert "Droppable options: EB, EBx2P" in result.output


def test_btv_fpp_drop_scenario_rejects_tp(tmp_path: Path) -> None:
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
            "--drop-scenario",
            "TP",
            "--out",
            str(tmp_path / "fpp_invalid_drop_tp.json"),
        ],
    )
    assert result.exit_code == 1
    assert "drop_scenario cannot include TP." in result.output


def test_btv_fpp_drop_scenario_override_only(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def _fake_build_cache_for_fpp(**_kwargs: Any) -> tuple[object, list[int]]:
        return object(), [14]

    def _fake_calculate_fpp(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {"fpp": 0.12, "nfpp": 0.01, "base_seed": 7}

    monkeypatch.setattr("tess_vetter.cli.fpp_cli._build_cache_for_fpp", _fake_build_cache_for_fpp)
    monkeypatch.setattr("tess_vetter.cli.fpp_cli.calculate_fpp", _fake_calculate_fpp)

    out_path = tmp_path / "fpp_drop_scenario_override_only.json"
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
            'drop_scenario=["EB","DEB"]',
            "--out",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["overrides"]["drop_scenario"] == ["EB", "DEB"]
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["provenance"]["runtime"]["overrides"]["drop_scenario"] == ["EB", "DEB"]


def test_btv_fpp_drop_scenario_explicit_flag_wins_over_override(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def _fake_build_cache_for_fpp(**_kwargs: Any) -> tuple[object, list[int]]:
        return object(), [14]

    def _fake_calculate_fpp(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {"fpp": 0.12, "nfpp": 0.01, "base_seed": 7}

    monkeypatch.setattr("tess_vetter.cli.fpp_cli._build_cache_for_fpp", _fake_build_cache_for_fpp)
    monkeypatch.setattr("tess_vetter.cli.fpp_cli.calculate_fpp", _fake_calculate_fpp)

    out_path = tmp_path / "fpp_drop_scenario_precedence.json"
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
            'drop_scenario=["DEB"]',
            "--drop-scenario",
            "EB",
            "--drop-scenario",
            "BEB",
            "--out",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen["overrides"]["drop_scenario"] == ["EB", "BEB"]
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["provenance"]["runtime"]["overrides"]["drop_scenario"] == ["EB", "BEB"]


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


def test_btv_fpp_detrend_implies_detrend_cache(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def _fake_execute_fpp(**kwargs: Any) -> tuple[dict[str, Any], list[int]]:
        seen.update(kwargs)
        return {"fpp": 0.12, "nfpp": 0.01, "base_seed": 7}, [14]

    monkeypatch.setattr("tess_vetter.cli.fpp_cli._execute_fpp", _fake_execute_fpp)

    out_path = tmp_path / "fpp_detrend_implies_cache.json"
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
    assert seen["detrend_cache"] is True

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    runtime = payload["provenance"]["runtime"]
    assert runtime["detrend_cache"] is True
    assert runtime["detrend_cache_requested"] is False


def test_btv_fpp_timeout_maps_to_exit_5(monkeypatch) -> None:
    def _fake_build_cache_for_fpp(**_kwargs: Any) -> tuple[object, list[int]]:
        return object(), [1]

    def _timeout(**_kwargs: Any) -> dict[str, Any]:
        raise TimeoutError("timed out")

    monkeypatch.setattr(
        "tess_vetter.cli.fpp_cli._build_cache_for_fpp",
        _fake_build_cache_for_fpp,
    )
    monkeypatch.setattr("tess_vetter.cli.fpp_cli.calculate_fpp", _timeout)

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
        "tess_vetter.cli.fpp_cli._resolve_candidate_inputs",
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

    monkeypatch.setattr("tess_vetter.cli.fpp_cli._resolve_candidate_inputs", _fake_resolve_candidate_inputs)
    monkeypatch.setattr("tess_vetter.cli.fpp_cli._build_cache_for_fpp", _fake_build_cache_for_fpp)
    monkeypatch.setattr("tess_vetter.cli.fpp_cli.calculate_fpp", _fake_calculate_fpp)
    monkeypatch.setattr("tess_vetter.cli.fpp_cli._estimate_detrended_depth_ppm", _fake_detrended_depth)

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

    monkeypatch.setattr("tess_vetter.cli.fpp_cli._resolve_candidate_inputs", _fake_resolve_candidate_inputs)
    monkeypatch.setattr("tess_vetter.cli.fpp_cli._build_cache_for_fpp", _fake_build_cache_for_fpp)
    monkeypatch.setattr("tess_vetter.cli.fpp_cli.calculate_fpp", _fake_calculate_fpp)
    monkeypatch.setattr("tess_vetter.cli.fpp_cli._estimate_detrended_depth_ppm", _fake_detrended_depth)

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

    monkeypatch.setattr("tess_vetter.cli.fpp_cli._build_cache_for_fpp", _fake_build_cache_for_fpp)
    monkeypatch.setattr("tess_vetter.cli.fpp_cli.calculate_fpp", _fake_calculate_fpp)
    monkeypatch.setattr("tess_vetter.cli.fpp_cli._load_auto_stellar_inputs", _fake_auto)

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
        "tess_vetter.cli.fpp_cli._build_cache_for_fpp",
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

        def get(self, key: str) -> object | None:
            return self.records.get(key)

        def keys(self) -> list[str]:
            return list(self.records.keys())

        def put(self, key: str, value: object) -> None:
            self.records[key] = value

    monkeypatch.setattr("tess_vetter.cli.fpp_cli.MASTClient", _FakeMASTClient)
    monkeypatch.setattr("tess_vetter.cli.fpp_cli.PersistentCache", _FakePersistentCache)

    from tess_vetter.cli.fpp_cli import _build_cache_for_fpp

    cache, loaded = _build_cache_for_fpp(tic_id=123, sectors=[14, 15], cache_dir=tmp_path)
    assert loaded == [14, 15]
    assert isinstance(cache, _FakePersistentCache)
    assert sorted(cache.records.keys()) == [
        "lc:123:14:pdcsap",
        "lc:123:15:pdcsap",
    ]


def test_btv_fpp_report_file_with_toi_prefers_toi_candidate_inputs(monkeypatch, tmp_path: Path) -> None:
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

    monkeypatch.setattr("tess_vetter.cli.fpp_cli._resolve_candidate_inputs", _fake_resolve_candidate_inputs)
    monkeypatch.setattr("tess_vetter.cli.fpp_cli._build_cache_for_fpp", _fake_build_cache_for_fpp)
    monkeypatch.setattr("tess_vetter.cli.fpp_cli.calculate_fpp", _fake_calculate_fpp)

    out_path = tmp_path / "fpp_report_with_toi_prefers_toi.json"
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
                "--network-ok",
                "--out",
                str(out_path),
            ],
        )

    assert result.exit_code == 0, result.output
    assert seen["resolve"]["toi"] == "TOI-123.01"
    assert seen["resolve"]["tic_id"] == 333
    assert seen["resolve"]["period_days"] is None
    assert seen["resolve"]["t0_btjd"] is None
    assert seen["resolve"]["duration_hours"] is None
    assert seen["resolve"]["depth_ppm"] == 900.0
    assert seen["build_cache"]["sectors"] == [20, 21]
    assert seen["fpp"]["sectors"] == [20, 21]

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["provenance"]["inputs"]["sectors"] == [20, 21]


def test_btv_fpp_report_file_only_supports_cli_report_schema(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    report_path = tmp_path / "report.summary_schema.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": "cli.report.v3",
                "report": {
                    "summary": {
                        "tic_id": 444,
                        "ephemeris": {
                            "period_days": 4.5,
                            "t0_btjd": 1400.125,
                            "duration_hours": 2.25,
                        },
                        "input_depth_ppm": 700.0,
                    },
                    "provenance": {"sectors_used": [11, 12]},
                },
            }
        ),
        encoding="utf-8",
    )

    def _fake_resolve_candidate_inputs(**kwargs: Any):
        seen["resolve"] = kwargs
        return 444, 4.5, 1400.125, 2.25, 700.0, {"source": "cli", "resolved_from": "cli"}

    def _fake_build_cache_for_fpp(**kwargs: Any) -> tuple[object, list[int]]:
        seen["build_cache"] = kwargs
        return object(), [11, 12]

    def _fake_calculate_fpp(**kwargs: Any) -> dict[str, Any]:
        seen["fpp"] = kwargs
        return {"fpp": 0.05, "nfpp": 0.005, "base_seed": 3}

    monkeypatch.setattr("tess_vetter.cli.fpp_cli._resolve_candidate_inputs", _fake_resolve_candidate_inputs)
    monkeypatch.setattr("tess_vetter.cli.fpp_cli._build_cache_for_fpp", _fake_build_cache_for_fpp)
    monkeypatch.setattr("tess_vetter.cli.fpp_cli.calculate_fpp", _fake_calculate_fpp)

    out_path = tmp_path / "fpp_report_only.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "fpp",
            "--report-file",
            str(report_path),
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Missing TIC identifier" not in result.output
    assert seen["resolve"]["toi"] is None
    assert seen["resolve"]["tic_id"] == 444
    assert seen["resolve"]["period_days"] == 4.5
    assert seen["resolve"]["t0_btjd"] == 1400.125
    assert seen["resolve"]["duration_hours"] == 2.25
    assert seen["resolve"]["depth_ppm"] == 700.0
    assert seen["build_cache"]["sectors"] == [11, 12]
    assert seen["fpp"]["sectors"] == [11, 12]

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["provenance"]["inputs"]["tic_id"] == 444


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

    monkeypatch.setattr("tess_vetter.cli.fpp_cli.MASTClient", _FakeMASTClient)
    monkeypatch.setattr(
        "tess_vetter.cli.fpp_cli._detrend_lightcurve_for_vetting",
        _fake_detrend_lightcurve_for_vetting,
    )
    monkeypatch.setattr("tess_vetter.cli.fpp_cli.measure_transit_depth", _fake_measure_transit_depth)
    monkeypatch.setattr("tess_vetter.cli.fpp_cli.calculate_fpp", _fake_calculate_fpp)

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
            "--cache-dir",
            str(tmp_path / "cache"),
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["network_calls"] >= 1
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

    monkeypatch.setattr("tess_vetter.cli.fpp_cli.MASTClient", _FakeMASTClient)
    monkeypatch.setattr(
        "tess_vetter.cli.fpp_cli._detrend_lightcurve_for_vetting",
        _fake_detrend_lightcurve_for_vetting,
    )
    monkeypatch.setattr("tess_vetter.cli.fpp_cli.measure_transit_depth", _fake_measure_transit_depth)
    monkeypatch.setattr("tess_vetter.cli.fpp_cli.calculate_fpp", _fake_calculate_fpp)

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
            "--cache-dir",
            str(tmp_path / "cache"),
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

    monkeypatch.setattr("tess_vetter.cli.fpp_cli._build_cache_for_fpp", _fake_build_cache_for_fpp)
    monkeypatch.setattr("tess_vetter.cli.fpp_cli.calculate_fpp", _fake_calculate_fpp)

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
    assert payload["schema_version"] == "cli.fpp.v3"


def test_runtime_artifacts_ready_true_when_cached_target_has_trilegal(monkeypatch, tmp_path: Path) -> None:
    from tess_vetter.cli import fpp_cli

    trilegal_path = tmp_path / "tri.csv"
    trilegal_path.write_text("a,b\n1,2\n", encoding="utf-8")

    class _Target:
        trilegal_fname = str(trilegal_path)

    monkeypatch.setattr(
        "tess_vetter.cli.fpp_cli.load_cached_triceratops_target",
        lambda **_kwargs: _Target(),
    )

    ready, details = fpp_cli._runtime_artifacts_ready(
        cache_dir=tmp_path,
        tic_id=123,
        sectors_loaded=[1, 2],
    )

    assert ready is True
    assert details["target_cached"] is True
    assert details["trilegal_cached"] is True
    assert details["trilegal_csv_path"] == str(trilegal_path)


def test_runtime_artifacts_ready_waits_for_prepare_lock(monkeypatch, tmp_path: Path) -> None:
    from tess_vetter.cli import fpp_cli

    trilegal_path = tmp_path / "tri_lock_wait.csv"
    trilegal_path.write_text("a,b\n1,2\n", encoding="utf-8")
    stage_state_path = tmp_path / "stage_state.json"
    stage_state_path.write_text(json.dumps({"status": "ok"}), encoding="utf-8")

    class _Target:
        trilegal_fname = str(trilegal_path)

    monkeypatch.setattr(
        "tess_vetter.cli.fpp_cli.load_cached_triceratops_target",
        lambda **_kwargs: _Target(),
    )

    result_box: dict[str, Any] = {}

    def _run_check() -> None:
        result_box["value"] = fpp_cli._runtime_artifacts_ready(
            cache_dir=tmp_path,
            tic_id=123,
            sectors_loaded=[1, 2],
            stage_state_path=stage_state_path,
        )

    with fpp_cli._triceratops_artifact_file_lock(
        cache_dir=tmp_path,
        tic_id=123,
        sectors_used=[1, 2],
        wait=True,
    ):
        worker = threading.Thread(target=_run_check, daemon=True)
        worker.start()
        time.sleep(0.2)
        assert worker.is_alive()

    worker.join(timeout=2.0)
    assert not worker.is_alive()
    ready, details = result_box["value"]
    assert ready is True
    assert details["stage_state_ok"] is True


def test_btv_fpp_run_require_prepared_fails_when_runtime_artifacts_missing(monkeypatch, tmp_path: Path) -> None:
    manifest = {
        "schema_version": "cli.fpp.prepare.v1",
        "created_at": "2026-01-01T00:00:00+00:00",
        "tic_id": 123,
        "period_days": 3.0,
        "t0_btjd": 1500.0,
        "duration_hours": 2.0,
        "depth_ppm_used": 500.0,
        "sectors_loaded": [10],
        "cache_dir": str(tmp_path / "cache"),
        "detrend": {},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr("tess_vetter.cli.fpp_cli._cache_missing_sectors", lambda **_kwargs: [])
    monkeypatch.setattr(
        "tess_vetter.cli.fpp_cli._runtime_artifacts_ready",
        lambda **_kwargs: (
            False,
            {"target_cached": False, "trilegal_cached": False, "trilegal_csv_path": None},
        ),
    )

    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "fpp-run",
            "--prepare-manifest",
            str(manifest_path),
            "--require-prepared",
            "--no-network",
            "-o",
            str(tmp_path / "fpp_run.json"),
        ],
    )

    assert result.exit_code == 4
    assert "Prepared runtime artifacts missing" in result.output


def test_btv_fpp_run_require_prepared_rejects_non_ok_stage_state(monkeypatch, tmp_path: Path) -> None:
    trilegal_path = tmp_path / "trilegal_ready.csv"
    trilegal_path.write_text("a,b\n1,2\n", encoding="utf-8")
    stage_state_path = tmp_path / "stage_state_failed.json"
    stage_state_path.write_text(json.dumps({"status": "failed"}), encoding="utf-8")

    manifest = {
        "schema_version": "cli.fpp.prepare.v1",
        "created_at": "2026-01-01T00:00:00+00:00",
        "tic_id": 123,
        "period_days": 3.0,
        "t0_btjd": 1500.0,
        "duration_hours": 2.0,
        "depth_ppm_used": 500.0,
        "sectors_loaded": [10],
        "cache_dir": str(tmp_path / "cache"),
        "detrend": {},
        "runtime_artifacts": {"stage_state_path": str(stage_state_path)},
    }
    manifest_path = tmp_path / "manifest_stage_failed.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    class _Target:
        trilegal_fname = str(trilegal_path)

    monkeypatch.setattr("tess_vetter.cli.fpp_cli._cache_missing_sectors", lambda **_kwargs: [])
    monkeypatch.setattr(
        "tess_vetter.cli.fpp_cli.load_cached_triceratops_target",
        lambda **_kwargs: _Target(),
    )

    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "fpp-run",
            "--prepare-manifest",
            str(manifest_path),
            "--require-prepared",
            "--no-network",
            "-o",
            str(tmp_path / "fpp_run_stage_failed.json"),
        ],
    )

    assert result.exit_code == 4
    assert "Prepared runtime artifacts missing" in result.output
    assert "stage_state_ok=False" in result.output


def test_btv_fpp_run_emits_replicate_progress(monkeypatch, tmp_path: Path) -> None:
    manifest = {
        "schema_version": "cli.fpp.prepare.v1",
        "created_at": "2026-01-01T00:00:00+00:00",
        "tic_id": 123,
        "period_days": 3.0,
        "t0_btjd": 1500.0,
        "duration_hours": 2.0,
        "depth_ppm_used": 500.0,
        "sectors_loaded": [10],
        "cache_dir": str(tmp_path / "cache"),
        "detrend": {},
    }
    manifest_path = tmp_path / "manifest_progress.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(
        "tess_vetter.cli.fpp_cli.resolve_stellar_inputs",
        lambda **_kwargs: ({}, {"source": "cli", "resolved_from": "cli"}),
    )

    def _fake_calculate_fpp(**kwargs: Any) -> dict[str, Any]:
        hook = kwargs.get("progress_hook")
        if callable(hook):
            hook({"event": "replicate_start", "replicate_index": 1, "replicates_total": 2, "seed": 101})
            hook(
                {
                    "event": "replicate_complete",
                    "replicate_index": 1,
                    "replicates_total": 2,
                    "seed": 101,
                    "status": "ok",
                }
            )
            hook({"event": "replicate_start", "replicate_index": 2, "replicates_total": 2, "seed": 102})
            hook(
                {
                    "event": "replicate_complete",
                    "replicate_index": 2,
                    "replicates_total": 2,
                    "seed": 102,
                    "status": "ok",
                }
            )
        return {"fpp": 0.01, "nfpp": 0.001, "base_seed": 101}

    monkeypatch.setattr("tess_vetter.cli.fpp_cli.calculate_fpp", _fake_calculate_fpp)

    runner = CliRunner()
    out_path = tmp_path / "fpp_run_progress.json"
    result = runner.invoke(
        enrich_cli.cli,
        [
            "fpp-run",
            "--prepare-manifest",
            str(manifest_path),
            "--allow-missing-prepared",
            "--replicates",
            "2",
            "--seed",
            "101",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "[fpp-run] replicate: 1/2 seed=101 start" in result.output
    assert "[fpp-run] replicate: 1/2 seed=101 status=ok" in result.output
    assert "[fpp-run] replicate: 2/2 seed=102 start" in result.output
    assert "[fpp-run] replicate: 2/2 seed=102 status=ok" in result.output


def test_btv_fpp_run_contrast_curve_fits_parsed_and_passed(monkeypatch, tmp_path: Path) -> None:
    fits = pytest.importorskip("astropy.io.fits")
    np = pytest.importorskip("numpy")
    seen: dict[str, Any] = {}

    manifest = {
        "schema_version": "cli.fpp.prepare.v1",
        "created_at": "2026-01-01T00:00:00+00:00",
        "tic_id": 123,
        "period_days": 3.0,
        "t0_btjd": 1500.0,
        "duration_hours": 2.0,
        "depth_ppm_used": 500.0,
        "sectors_loaded": [10],
        "cache_dir": str(tmp_path / "cache"),
        "detrend": {},
    }
    manifest_path = tmp_path / "manifest_fits_cc.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    cc_path = tmp_path / "TIC149302744I-at20190714_soarspeckle.fits"
    rng = np.random.default_rng(7)
    image = rng.normal(0.0, 0.001, size=(200, 200))
    image[100, 100] += 2.0
    fits.PrimaryHDU(data=image).writeto(cc_path)

    monkeypatch.setattr("tess_vetter.cli.fpp_cli._cache_missing_sectors", lambda **_kwargs: [])
    monkeypatch.setattr(
        "tess_vetter.cli.fpp_cli._runtime_artifacts_ready",
        lambda **_kwargs: (True, {"target_cached": True, "trilegal_cached": True, "trilegal_csv_path": "ok.csv"}),
    )
    monkeypatch.setattr(
        "tess_vetter.cli.fpp_cli.resolve_stellar_inputs",
        lambda **_kwargs: ({}, {"source": "cli", "resolved_from": "cli"}),
    )

    def _fake_calculate_fpp(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        return {"fpp": 0.01, "nfpp": 0.001, "base_seed": 101}

    monkeypatch.setattr("tess_vetter.cli.fpp_cli.calculate_fpp", _fake_calculate_fpp)

    out_path = tmp_path / "fpp_run_with_fits_cc.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "fpp-run",
            "--prepare-manifest",
            str(manifest_path),
            "--require-prepared",
            "--contrast-curve",
            str(cc_path),
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    cc = seen["contrast_curve"]
    assert cc is not None
    assert len(cc.separation_arcsec) >= 2
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["provenance"]["contrast_curve"]["parse_provenance"]["strategy"] == "fits_image_azimuthal"


def test_btv_fpp_run_can_execute_twice_with_same_manifest(monkeypatch, tmp_path: Path) -> None:
    manifest = {
        "schema_version": "cli.fpp.prepare.v1",
        "created_at": "2026-01-01T00:00:00+00:00",
        "tic_id": 123,
        "period_days": 3.0,
        "t0_btjd": 1500.0,
        "duration_hours": 2.0,
        "depth_ppm_used": 500.0,
        "sectors_loaded": [10],
        "cache_dir": str(tmp_path / "cache"),
        "detrend": {},
    }
    manifest_path = tmp_path / "manifest_twice.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    seen: dict[str, int] = {"calls": 0}

    monkeypatch.setattr(
        "tess_vetter.cli.fpp_cli.resolve_stellar_inputs",
        lambda **_kwargs: ({}, {"source": "cli", "resolved_from": "cli"}),
    )
    monkeypatch.setattr("tess_vetter.cli.fpp_cli._cache_missing_sectors", lambda **_kwargs: [])
    monkeypatch.setattr(
        "tess_vetter.cli.fpp_cli._runtime_artifacts_ready",
        lambda **_kwargs: (True, {"target_cached": True, "trilegal_cached": True, "trilegal_csv_path": "ok.csv"}),
    )

    def _fake_calculate_fpp(**_kwargs: Any) -> dict[str, Any]:
        seen["calls"] += 1
        return {"fpp": 0.01, "nfpp": 0.001, "base_seed": 101}

    monkeypatch.setattr("tess_vetter.cli.fpp_cli.calculate_fpp", _fake_calculate_fpp)

    runner = CliRunner()
    out1 = tmp_path / "fpp_run_1.json"
    out2 = tmp_path / "fpp_run_2.json"
    result1 = runner.invoke(
        enrich_cli.cli,
        [
            "fpp-run",
            "--prepare-manifest",
            str(manifest_path),
            "--require-prepared",
            "--out",
            str(out1),
        ],
    )
    result2 = runner.invoke(
        enrich_cli.cli,
        [
            "fpp-run",
            "--prepare-manifest",
            str(manifest_path),
            "--require-prepared",
            "--out",
            str(out2),
        ],
    )

    assert result1.exit_code == 0, result1.output
    assert result2.exit_code == 0, result2.output
    assert seen["calls"] == 2


def test_btv_fpp_prepare_supports_short_o(monkeypatch, tmp_path: Path) -> None:
    from tess_vetter.platform.io import PersistentCache

    def _fake_resolve_candidate_inputs(**_kwargs: Any):
        return (123, 7.5, 2500.25, 3.0, 900.0, {"source": "cli", "resolved_from": "cli"})

    def _fake_build_cache_for_fpp(**_kwargs: Any):
        cache = PersistentCache(cache_dir=tmp_path / "cache")
        return cache, [14, 15]

    monkeypatch.setattr(
        "tess_vetter.cli.fpp_cli._resolve_candidate_inputs",
        _fake_resolve_candidate_inputs,
    )
    monkeypatch.setattr(
        "tess_vetter.cli.fpp_cli._build_cache_for_fpp",
        _fake_build_cache_for_fpp,
    )

    out_path = tmp_path / "prepare_manifest.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "fpp-prepare",
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
            "--no-network",
            "-o",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "cli.fpp.prepare.v1"


def test_btv_fpp_prepare_retries_after_transient_failure_with_failed_state_file_present(
    monkeypatch, tmp_path: Path
) -> None:
    from tess_vetter.platform.io import PersistentCache

    def _fake_resolve_candidate_inputs(**_kwargs: Any):
        return (123, 7.5, 2500.25, 3.0, 900.0, {"source": "cli", "resolved_from": "cli"})

    def _fake_build_cache_for_fpp(**_kwargs: Any):
        cache = PersistentCache(cache_dir=tmp_path / "cache")
        return cache, [14, 15]

    calls = {"n": 0}

    def _fake_stage_runtime(**_kwargs: Any):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("temporary network timeout")
        return {
            "trilegal_csv_path": str(tmp_path / "cache" / "triceratops" / "123_TRILEGAL.csv"),
            "target_cache_hit": True,
            "trilegal_cache_hit": False,
            "runtime_seconds": 0.5,
        }

    monkeypatch.setattr(
        "tess_vetter.cli.fpp_cli._resolve_candidate_inputs",
        _fake_resolve_candidate_inputs,
    )
    monkeypatch.setattr(
        "tess_vetter.cli.fpp_cli._build_cache_for_fpp",
        _fake_build_cache_for_fpp,
    )
    monkeypatch.setattr(
        "tess_vetter.cli.fpp_cli.stage_triceratops_runtime_artifacts",
        _fake_stage_runtime,
    )

    runner = CliRunner()
    out1 = tmp_path / "prepare_fail.json"
    first = runner.invoke(
        enrich_cli.cli,
        [
            "fpp-prepare",
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
            "--network-ok",
            "--out",
            str(out1),
        ],
    )
    assert first.exit_code != 0

    # Simulate stale failed stage-state file that should not block retry.
    stage_state = (
        tmp_path / "cache" / "triceratops" / "staging_state" / "tic_123__sectors_14-15.json"
    )
    stage_state.parent.mkdir(parents=True, exist_ok=True)
    stage_state.write_text(
        json.dumps(
            {
                "status": "failed",
                "stage": "trilegal_prefetch",
                "error_code": "NetworkTimeoutError",
                "error": "temporary network timeout",
            }
        ),
        encoding="utf-8",
    )

    out2 = tmp_path / "prepare_retry.json"
    second = runner.invoke(
        enrich_cli.cli,
        [
            "fpp-prepare",
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
            "--network-ok",
            "--out",
            str(out2),
        ],
    )

    assert second.exit_code == 0, second.output
    assert calls["n"] == 2


def test_btv_fpp_prepare_passes_timeout_seconds_to_runtime_staging(
    monkeypatch, tmp_path: Path
) -> None:
    seen: dict[str, Any] = {}

    from tess_vetter.platform.io import PersistentCache

    def _fake_resolve_candidate_inputs(**_kwargs: Any):
        return (123, 7.5, 2500.25, 3.0, 900.0, {"source": "cli", "resolved_from": "cli"})

    def _fake_build_cache_for_fpp(**_kwargs: Any):
        cache = PersistentCache(cache_dir=tmp_path / "cache")
        return cache, [14, 15]

    def _fake_stage_runtime(**kwargs: Any):
        seen["timeout_seconds"] = kwargs.get("timeout_seconds")
        return {
            "trilegal_csv_path": str(tmp_path / "cache" / "triceratops" / "123_TRILEGAL.csv"),
            "target_cache_hit": True,
            "trilegal_cache_hit": False,
            "runtime_seconds": 0.5,
        }

    monkeypatch.setattr(
        "tess_vetter.cli.fpp_cli._resolve_candidate_inputs",
        _fake_resolve_candidate_inputs,
    )
    monkeypatch.setattr(
        "tess_vetter.cli.fpp_cli._build_cache_for_fpp",
        _fake_build_cache_for_fpp,
    )
    monkeypatch.setattr(
        "tess_vetter.cli.fpp_cli.stage_triceratops_runtime_artifacts",
        _fake_stage_runtime,
    )

    out_path = tmp_path / "prepare_timeout_manifest.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "fpp-prepare",
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
            "--network-ok",
            "--timeout-seconds",
            "900",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["timeout_seconds"] == 900.0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["runtime_artifacts"]["timeout_seconds_requested"] == 900.0


def test_btv_fpp_prepare_with_toi_prefers_toi_candidate_inputs(monkeypatch, tmp_path: Path) -> None:
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

    from tess_vetter.platform.io import PersistentCache

    def _fake_resolve_candidate_inputs(**kwargs: Any):
        seen["resolve"] = kwargs
        return (333, 3.1, 1400.5, 2.2, 900.0, {"source": "toi", "resolved_from": "toi"})

    def _fake_build_cache_for_fpp(**kwargs: Any):
        seen["build_cache"] = kwargs
        cache = PersistentCache(cache_dir=tmp_path / "cache")
        return cache, [20, 21]

    monkeypatch.setattr(
        "tess_vetter.cli.fpp_cli._resolve_candidate_inputs",
        _fake_resolve_candidate_inputs,
    )
    monkeypatch.setattr(
        "tess_vetter.cli.fpp_cli._build_cache_for_fpp",
        _fake_build_cache_for_fpp,
    )
    monkeypatch.setattr(
        "tess_vetter.cli.fpp_cli.stage_triceratops_runtime_artifacts",
        lambda **_kwargs: {
            "trilegal_csv_path": str(tmp_path / "cache" / "triceratops" / "fake.csv"),
            "target_cache_hit": True,
            "trilegal_cache_hit": False,
            "runtime_seconds": 1.0,
        },
    )

    out_path = tmp_path / "prepare_with_toi_manifest.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "fpp-prepare",
            "--toi",
            "TOI-123.01",
            "--report-file",
            str(report_path),
            "--tic-id",
            "333",
            "--depth-ppm",
            "900.0",
            "--network-ok",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["resolve"]["toi"] == "TOI-123.01"
    assert seen["resolve"]["tic_id"] == 333
    assert seen["resolve"]["period_days"] is None
    assert seen["resolve"]["t0_btjd"] is None
    assert seen["resolve"]["duration_hours"] is None
    assert seen["resolve"]["depth_ppm"] == 900.0
    assert seen["build_cache"]["sectors"] == [20, 21]


def test_btv_fpp_supports_prepare_manifest_mode(monkeypatch, tmp_path: Path) -> None:
    manifest = {
        "schema_version": "cli.fpp.prepare.v1",
        "created_at": "2026-01-01T00:00:00+00:00",
        "tic_id": 123,
        "period_days": 3.0,
        "t0_btjd": 1500.0,
        "duration_hours": 2.0,
        "depth_ppm_used": 500.0,
        "sectors_loaded": [10],
        "cache_dir": str(tmp_path / "cache"),
        "detrend": {},
    }
    manifest_path = tmp_path / "manifest_for_fpp.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(
        "tess_vetter.cli.fpp_cli.resolve_stellar_inputs",
        lambda **_kwargs: ({}, {"source": "cli", "resolved_from": "cli"}),
    )
    monkeypatch.setattr("tess_vetter.cli.fpp_cli._cache_missing_sectors", lambda **_kwargs: [])
    monkeypatch.setattr(
        "tess_vetter.cli.fpp_cli._runtime_artifacts_ready",
        lambda **_kwargs: (True, {"target_cached": True, "trilegal_cached": True, "trilegal_csv_path": "ok.csv"}),
    )
    monkeypatch.setattr(
        "tess_vetter.cli.fpp_cli.calculate_fpp",
        lambda **_kwargs: {"fpp": 0.01, "nfpp": 0.001, "base_seed": 101},
    )

    out_path = tmp_path / "fpp_from_manifest.json"
    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "fpp",
            "--prepare-manifest",
            str(manifest_path),
            "--require-prepared",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "cli.fpp.v3"
    assert payload["provenance"]["prepare_manifest"]["path"] == str(manifest_path)


def test_btv_fpp_prepare_manifest_rejects_conflicting_candidate_options(tmp_path: Path) -> None:
    manifest = {
        "schema_version": "cli.fpp.prepare.v1",
        "created_at": "2026-01-01T00:00:00+00:00",
        "tic_id": 123,
        "period_days": 3.0,
        "t0_btjd": 1500.0,
        "duration_hours": 2.0,
        "depth_ppm_used": 500.0,
        "sectors_loaded": [10],
        "cache_dir": str(tmp_path / "cache"),
        "detrend": {},
    }
    manifest_path = tmp_path / "manifest_conflict.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "fpp",
            "--prepare-manifest",
            str(manifest_path),
            "--tic-id",
            "123",
        ],
    )

    assert result.exit_code != 0
    assert "--prepare-manifest cannot be combined" in result.output
