from __future__ import annotations

import json
from typing import Any

import numpy as np
from click.testing import CliRunner

import bittr_tess_vetter.cli.enrich_cli as enrich_cli


class _FakeLightCurve:
    def __init__(self) -> None:
        self.time = np.linspace(0.0, 10.0, 200, dtype=np.float64)
        self.flux = np.ones(200, dtype=np.float64)
        self.flux_err = np.full(200, 1e-4, dtype=np.float64)
        self.sector = 82


class _FakeMASTClient:
    def download_all_sectors(self, *args: Any, **kwargs: Any) -> list[_FakeLightCurve]:
        return [_FakeLightCurve()]


class _FakeSweepResult:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "stable": True,
            "metric_variance": 0.05,
            "score_spread_iqr_over_median": 0.03,
            "depth_spread_iqr_over_median": 0.05,
            "n_variants_total": len(self._rows),
            "n_variants_ok": len(self._rows),
            "n_variants_failed": 0,
            "best_variant_id": "variant_a",
            "worst_variant_id": "variant_b",
            "stability_threshold": 0.2,
            "notes": [],
            "sweep_table": self._rows,
        }


def _base_args() -> list[str]:
    return [
        "detrend-grid",
        "--tic-id",
        "123",
        "--period-days",
        "10.5",
        "--t0-btjd",
        "2000.2",
        "--duration-hours",
        "2.5",
    ]


def _json_from_cli_output(output: str) -> dict[str, Any]:
    start = output.find("{")
    end = output.rfind("}")
    assert start >= 0 and end >= start, output
    return json.loads(output[start : end + 1])


def test_detrend_grid_command_presence_in_root_help() -> None:
    runner = CliRunner()
    result = runner.invoke(enrich_cli.cli, ["--help"])
    assert result.exit_code == 0, result.output
    assert "detrend-grid" in result.output


def test_detrend_grid_output_schema_keys(monkeypatch) -> None:
    rows = [
        {
            "variant_id": "variant_a",
            "status": "ok",
            "backend": "numpy",
            "runtime_seconds": 0.01,
            "n_points_used": 200,
            "downsample_factor": 1,
            "outlier_policy": "none",
            "detrender": "none",
            "score": 5.0,
            "depth_hat_ppm": 250.0,
            "depth_err_ppm": 10.0,
            "warnings": [],
            "failure_reason": None,
            "variant_config": {"variant_id": "variant_a"},
            "gp_hyperparams": None,
            "gp_fit_diagnostics": None,
        },
        {
            "variant_id": "variant_b",
            "status": "ok",
            "backend": "numpy",
            "runtime_seconds": 0.01,
            "n_points_used": 200,
            "downsample_factor": 2,
            "outlier_policy": "sigma_clip_4",
            "detrender": "running_median_0.5d",
            "score": 4.5,
            "depth_hat_ppm": 240.0,
            "depth_err_ppm": 11.0,
            "warnings": [],
            "failure_reason": None,
            "variant_config": {"variant_id": "variant_b"},
            "gp_hyperparams": None,
            "gp_fit_diagnostics": None,
        },
    ]

    monkeypatch.setattr("bittr_tess_vetter.cli.detrend_grid_cli.MASTClient", _FakeMASTClient)
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.detrend_grid_cli.compute_sensitivity_sweep_numpy",
        lambda **_kwargs: _FakeSweepResult(rows),
    )

    runner = CliRunner()
    result = runner.invoke(enrich_cli.cli, _base_args())
    assert result.exit_code == 0, result.output
    payload = _json_from_cli_output(result.output)

    assert payload["schema_version"] == "cli.detrend_grid.v1"
    core_keys = {
        "stable",
        "metric_variance",
        "n_variants_total",
        "n_variants_ok",
        "n_variants_failed",
        "best_variant_id",
        "worst_variant_id",
        "stability_threshold",
        "notes",
        "sweep_table",
    }
    assert core_keys.issubset(payload.keys())
    assert "best_variant" in payload
    assert "recommended_next_step" in payload
    assert "recommended_detrend_flags" in payload
    assert "result_summary" in payload
    assert "provenance" in payload
    assert "ranked_sweep_table" in payload
    assert payload["verdict"] == "STABLE"
    assert payload["verdict_source"] == "$.stable"
    assert payload["result"]["verdict"] == payload["verdict"]
    assert payload["result"]["verdict_source"] == payload["verdict_source"]
    assert payload["result"]["recommended_detrend_flags"] is None
    assert payload["best_variant"]["variant_id"] == "variant_a"
    assert payload["result_summary"]["best_variant_id"] == "variant_a"
    assert payload["result_summary"]["best_variant_rank"] == 1
    assert payload["result_summary"]["best_variant_config"] == {"variant_id": "variant_a"}
    assert payload["result_summary"]["recommended_next_step"] is None
    assert payload["result_summary"]["recommended_detrend_flags"] is None
    assert payload["result_summary"]["stable"] is True
    assert payload["result_summary"]["n_variants_total"] == 2
    assert payload["result_summary"]["depth_hat_ppm_range"] == {"min": 240.0, "max": 250.0}
    assert payload["provenance"]["command"] == "detrend-grid"
    assert payload["provenance"]["effective_grid_config"]["downsample_levels"] == [1, 2, 5]
    assert payload["provenance"]["effective_grid_config"]["outlier_policies"] == ["none", "sigma_clip_4"]
    assert payload["provenance"]["effective_grid_config"]["detrenders"] == [
        "none",
        "running_median_0.5d",
        "transit_masked_bin_median",
    ]
    assert payload["provenance"]["effective_grid_config"]["transit_masked_bin_median"] == {
        "bin_hours": [4.0, 6.0, 8.0],
        "buffer_factor": [1.5, 2.0, 3.0],
        "sigma_clip": [3.0, 5.0],
    }
    assert payload["variant_axes"]["detrenders"] == [
        "none",
        "running_median_0.5d",
        "transit_masked_bin_median",
    ]
    assert payload["variant_axes"]["transit_masked_bin_median"] == {
        "bin_hours": [4.0, 6.0, 8.0],
        "buffer_factor": [1.5, 2.0, 3.0],
        "sigma_clip": [3.0, 5.0],
    }
    assert payload["provenance"]["variant_counts"]["cross_product"] == 120
    assert payload["sweep_table"][0]["depth_method"] == "template_ls"
    assert "optimized for variant ranking" in payload["sweep_table"][0]["depth_note"]


def test_detrend_grid_accepts_short_o_alias(monkeypatch, tmp_path) -> None:
    rows = [
        {
            "variant_id": "variant_a",
            "status": "ok",
            "backend": "numpy",
            "runtime_seconds": 0.01,
            "n_points_used": 200,
            "downsample_factor": 1,
            "outlier_policy": "none",
            "detrender": "none",
            "score": 5.0,
            "depth_hat_ppm": 250.0,
            "depth_err_ppm": 10.0,
            "warnings": [],
            "failure_reason": None,
            "variant_config": {"variant_id": "variant_a"},
            "gp_hyperparams": None,
            "gp_fit_diagnostics": None,
        }
    ]

    monkeypatch.setattr("bittr_tess_vetter.cli.detrend_grid_cli.MASTClient", _FakeMASTClient)
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.detrend_grid_cli.compute_sensitivity_sweep_numpy",
        lambda **_kwargs: _FakeSweepResult(rows),
    )

    out_path = tmp_path / "detrend_grid_short_o.json"
    runner = CliRunner()
    result = runner.invoke(enrich_cli.cli, [*_base_args(), "-o", str(out_path)])
    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["best_variant"]["variant_id"] == "variant_a"


def test_detrend_grid_emits_progress_to_stderr(monkeypatch) -> None:
    rows = [
        {
            "variant_id": "variant_a",
            "status": "ok",
            "backend": "numpy",
            "runtime_seconds": 0.01,
            "n_points_used": 200,
            "downsample_factor": 1,
            "outlier_policy": "none",
            "detrender": "none",
            "score": 5.0,
            "depth_hat_ppm": 250.0,
            "depth_err_ppm": 10.0,
            "warnings": [],
            "failure_reason": None,
            "variant_config": {"variant_id": "variant_a"},
            "gp_hyperparams": None,
            "gp_fit_diagnostics": None,
        }
    ]
    monkeypatch.setattr("bittr_tess_vetter.cli.detrend_grid_cli.MASTClient", _FakeMASTClient)
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.detrend_grid_cli.compute_sensitivity_sweep_numpy",
        lambda **_kwargs: _FakeSweepResult(rows),
    )

    runner = CliRunner()
    result = runner.invoke(enrich_cli.cli, _base_args())
    assert result.exit_code == 0, result.output
    assert "[detrend-grid] start" in result.output
    assert "[detrend-grid] completed" in result.output


def test_detrend_grid_emits_recommended_next_step_for_transit_masked_best(monkeypatch) -> None:
    rows = [
        {
            "variant_id": "best_tm",
            "status": "ok",
            "backend": "numpy",
            "runtime_seconds": 0.01,
            "n_points_used": 200,
            "downsample_factor": 1,
            "outlier_policy": "none",
            "detrender": "transit_masked_bin_median",
            "score": 9.0,
            "depth_hat_ppm": 215.0,
            "depth_err_ppm": 10.0,
            "warnings": [],
            "failure_reason": None,
            "variant_config": {
                "variant_id": "best_tm",
                "detrender": "transit_masked_bin_median",
                "detrender_bin_hours": 6.0,
                "detrender_buffer_factor": 2.0,
                "detrender_sigma_clip": 5.0,
            },
            "gp_hyperparams": None,
            "gp_fit_diagnostics": None,
        }
    ]
    monkeypatch.setattr("bittr_tess_vetter.cli.detrend_grid_cli.MASTClient", _FakeMASTClient)
    class _BestTransitSweep:
        def to_dict(self) -> dict[str, Any]:
            return {
                "stable": True,
                "metric_variance": 0.05,
                "score_spread_iqr_over_median": 0.03,
                "depth_spread_iqr_over_median": 0.05,
                "n_variants_total": len(rows),
                "n_variants_ok": len(rows),
                "n_variants_failed": 0,
                "best_variant_id": "best_tm",
                "worst_variant_id": "best_tm",
                "stability_threshold": 0.2,
                "notes": [],
                "sweep_table": rows,
            }

    monkeypatch.setattr(
        "bittr_tess_vetter.cli.detrend_grid_cli.compute_sensitivity_sweep_numpy",
        lambda **_kwargs: _BestTransitSweep(),
    )

    runner = CliRunner()
    result = runner.invoke(enrich_cli.cli, _base_args())
    assert result.exit_code == 0, result.output
    payload = _json_from_cli_output(result.output)
    assert (
        payload["recommended_next_step"]
        == "btv vet --detrend transit_masked_bin_median --detrend-bin-hours 6 --detrend-buffer 2 --detrend-sigma-clip 5"
    )
    assert payload["recommended_detrend_flags"] == {
        "detrend": "transit_masked_bin_median",
        "bin_hours": 6.0,
        "buffer": 2.0,
        "sigma_clip": 5.0,
    }
    assert payload["result"]["recommended_detrend_flags"] == payload["recommended_detrend_flags"]
    assert payload["result_summary"]["recommended_next_step"] == payload["recommended_next_step"]
    assert payload["result_summary"]["recommended_detrend_flags"] == payload["recommended_detrend_flags"]


def test_detrend_grid_seed_is_deterministic_and_defaults_to_zero(monkeypatch) -> None:
    monkeypatch.setattr("bittr_tess_vetter.cli.detrend_grid_cli.MASTClient", _FakeMASTClient)

    def _fake_sweep(**kwargs: Any) -> _FakeSweepResult:
        seed = int(kwargs["random_seed"])
        rng = np.random.default_rng(seed)
        score_a = float(rng.uniform(1.0, 10.0))
        score_b = float(rng.uniform(1.0, 10.0))
        rows = [
            {
                "variant_id": "variant_a",
                "status": "ok",
                "backend": "numpy",
                "runtime_seconds": 0.01,
                "n_points_used": 200,
                "downsample_factor": 1,
                "outlier_policy": "none",
                "detrender": "none",
                "score": score_a,
                "depth_hat_ppm": 250.0,
                "depth_err_ppm": 10.0,
                "warnings": [],
                "failure_reason": None,
                "variant_config": {"variant_id": "variant_a"},
                "gp_hyperparams": None,
                "gp_fit_diagnostics": None,
            },
            {
                "variant_id": "variant_b",
                "status": "ok",
                "backend": "numpy",
                "runtime_seconds": 0.01,
                "n_points_used": 200,
                "downsample_factor": 2,
                "outlier_policy": "sigma_clip_4",
                "detrender": "running_median_0.5d",
                "score": score_b,
                "depth_hat_ppm": 240.0,
                "depth_err_ppm": 11.0,
                "warnings": [],
                "failure_reason": None,
                "variant_config": {"variant_id": "variant_b"},
                "gp_hyperparams": None,
                "gp_fit_diagnostics": None,
            },
        ]
        best_variant_id = "variant_a" if abs(score_a) >= abs(score_b) else "variant_b"
        result = _FakeSweepResult(rows)
        result.to_dict = lambda: {
            "stable": True,
            "metric_variance": 0.05,
            "score_spread_iqr_over_median": 0.03,
            "depth_spread_iqr_over_median": 0.05,
            "n_variants_total": len(rows),
            "n_variants_ok": len(rows),
            "n_variants_failed": 0,
            "best_variant_id": best_variant_id,
            "worst_variant_id": "variant_b" if best_variant_id == "variant_a" else "variant_a",
            "stability_threshold": 0.2,
            "notes": [],
            "sweep_table": rows,
        }
        return result

    monkeypatch.setattr(
        "bittr_tess_vetter.cli.detrend_grid_cli.compute_sensitivity_sweep_numpy",
        _fake_sweep,
    )

    runner = CliRunner()
    result_a = runner.invoke(enrich_cli.cli, [*_base_args(), "--random-seed", "17"])
    result_b = runner.invoke(enrich_cli.cli, [*_base_args(), "--random-seed", "17"])
    result_default = runner.invoke(enrich_cli.cli, _base_args())
    result_zero = runner.invoke(enrich_cli.cli, [*_base_args(), "--random-seed", "0"])

    assert result_a.exit_code == 0, result_a.output
    assert result_b.exit_code == 0, result_b.output
    assert result_default.exit_code == 0, result_default.output
    assert result_zero.exit_code == 0, result_zero.output

    payload_a = _json_from_cli_output(result_a.output)
    payload_b = _json_from_cli_output(result_b.output)
    payload_default = _json_from_cli_output(result_default.output)
    payload_zero = _json_from_cli_output(result_zero.output)

    assert payload_a["best_variant"]["variant_id"] == payload_b["best_variant"]["variant_id"]
    assert payload_a["ranked_sweep_table"][0]["variant_id"] == payload_b["ranked_sweep_table"][0]["variant_id"]
    assert payload_default["best_variant"]["variant_id"] == payload_zero["best_variant"]["variant_id"]
    assert payload_default["provenance"]["random_seed"] == 0


def test_detrend_grid_passes_cache_dir_to_mast_client(monkeypatch, tmp_path) -> None:
    seen: dict[str, Any] = {}

    class _CacheAwareFakeMASTClient:
        def __init__(self, cache_dir: str | None = None) -> None:
            seen["cache_dir"] = cache_dir

        def download_all_sectors(self, *args: Any, **kwargs: Any) -> list[_FakeLightCurve]:
            _ = args, kwargs
            return [_FakeLightCurve()]

    rows = [
        {
            "variant_id": "variant_a",
            "status": "ok",
            "backend": "numpy",
            "runtime_seconds": 0.01,
            "n_points_used": 200,
            "downsample_factor": 1,
            "outlier_policy": "none",
            "detrender": "none",
            "score": 5.0,
            "depth_hat_ppm": 250.0,
            "depth_err_ppm": 10.0,
            "warnings": [],
            "failure_reason": None,
            "variant_config": {"variant_id": "variant_a"},
            "gp_hyperparams": None,
            "gp_fit_diagnostics": None,
        }
    ]
    monkeypatch.setattr("bittr_tess_vetter.cli.detrend_grid_cli.MASTClient", _CacheAwareFakeMASTClient)
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.detrend_grid_cli.compute_sensitivity_sweep_numpy",
        lambda **_kwargs: _FakeSweepResult(rows),
    )

    cache_dir = tmp_path / "mast_cache"
    runner = CliRunner()
    result = runner.invoke(enrich_cli.cli, [*_base_args(), "--cache-dir", str(cache_dir)])
    assert result.exit_code == 0, result.output
    assert seen["cache_dir"] == str(cache_dir)


def test_detrend_grid_adds_check_resolution_note_for_v16_model_competition(monkeypatch, tmp_path) -> None:
    rows = [
        {
            "variant_id": "variant_a",
            "status": "ok",
            "backend": "numpy",
            "runtime_seconds": 0.01,
            "n_points_used": 200,
            "downsample_factor": 1,
            "outlier_policy": "none",
            "detrender": "none",
            "score": 5.0,
            "depth_hat_ppm": 250.0,
            "depth_err_ppm": 10.0,
            "warnings": [],
            "failure_reason": None,
            "variant_config": {"variant_id": "variant_a"},
            "gp_hyperparams": None,
            "gp_fit_diagnostics": None,
        }
    ]
    vet_payload = {
        "summary": {
            "concerns": ["MODEL_PREFERS_NON_TRANSIT"],
            "disposition_hint": "needs_model_competition_review",
        }
    }
    vet_path = tmp_path / "vet.json"
    vet_path.write_text(json.dumps(vet_payload), encoding="utf-8")

    monkeypatch.setattr("bittr_tess_vetter.cli.detrend_grid_cli.MASTClient", _FakeMASTClient)
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.detrend_grid_cli.compute_sensitivity_sweep_numpy",
        lambda **_kwargs: _FakeSweepResult(rows),
    )

    runner = CliRunner()
    result = runner.invoke(enrich_cli.cli, [*_base_args(), "--vet-summary-path", str(vet_path)])
    assert result.exit_code == 0, result.output
    payload = _json_from_cli_output(result.output)

    assert payload["check_resolution_note"]["check_id"] == "V16"
    assert payload["check_resolution_note"]["reason"] == "model_competition_concern"
    assert "MODEL_PREFERS_NON_TRANSIT" in payload["check_resolution_note"]["triggers"]["concerns"]
    assert payload["provenance"]["check_resolution_note"]["check_id"] == "V16"
    assert payload["provenance"]["vet_summary"]["source_path"] == str(vet_path)


def test_detrend_grid_omits_check_resolution_note_without_vet_summary(monkeypatch) -> None:
    rows = [
        {
            "variant_id": "variant_a",
            "status": "ok",
            "backend": "numpy",
            "runtime_seconds": 0.01,
            "n_points_used": 200,
            "downsample_factor": 1,
            "outlier_policy": "none",
            "detrender": "none",
            "score": 5.0,
            "depth_hat_ppm": 250.0,
            "depth_err_ppm": 10.0,
            "warnings": [],
            "failure_reason": None,
            "variant_config": {"variant_id": "variant_a"},
            "gp_hyperparams": None,
            "gp_fit_diagnostics": None,
        }
    ]
    monkeypatch.setattr("bittr_tess_vetter.cli.detrend_grid_cli.MASTClient", _FakeMASTClient)
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.detrend_grid_cli.compute_sensitivity_sweep_numpy",
        lambda **_kwargs: _FakeSweepResult(rows),
    )

    runner = CliRunner()
    result = runner.invoke(enrich_cli.cli, _base_args())
    assert result.exit_code == 0, result.output
    payload = _json_from_cli_output(result.output)

    assert "check_resolution_note" not in payload


def test_detrend_grid_accepts_summary_at_payload_root(monkeypatch, tmp_path) -> None:
    rows = [
        {
            "variant_id": "variant_a",
            "status": "ok",
            "backend": "numpy",
            "runtime_seconds": 0.01,
            "n_points_used": 200,
            "downsample_factor": 1,
            "outlier_policy": "none",
            "detrender": "none",
            "score": 5.0,
            "depth_hat_ppm": 250.0,
            "depth_err_ppm": 10.0,
            "warnings": [],
            "failure_reason": None,
            "variant_config": {"variant_id": "variant_a"},
            "gp_hyperparams": None,
            "gp_fit_diagnostics": None,
        }
    ]
    vet_summary_root = {
        "concerns": ["MODEL_PREFERS_NON_TRANSIT"],
        "disposition_hint": "needs_model_competition_review",
    }
    vet_path = tmp_path / "vet_summary_root.json"
    vet_path.write_text(json.dumps(vet_summary_root), encoding="utf-8")

    monkeypatch.setattr("bittr_tess_vetter.cli.detrend_grid_cli.MASTClient", _FakeMASTClient)
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.detrend_grid_cli.compute_sensitivity_sweep_numpy",
        lambda **_kwargs: _FakeSweepResult(rows),
    )

    runner = CliRunner()
    result = runner.invoke(enrich_cli.cli, [*_base_args(), "--vet-summary-path", str(vet_path)])
    assert result.exit_code == 0, result.output
    payload = _json_from_cli_output(result.output)
    assert payload["check_resolution_note"]["check_id"] == "V16"
    assert payload["provenance"]["vet_summary"]["summary_source"] == "payload_root"


def test_detrend_grid_supports_report_file_inputs(monkeypatch, tmp_path) -> None:
    rows = [
        {
            "variant_id": "variant_a",
            "status": "ok",
            "backend": "numpy",
            "runtime_seconds": 0.01,
            "n_points_used": 200,
            "downsample_factor": 1,
            "outlier_policy": "none",
            "detrender": "none",
            "score": 5.0,
            "depth_hat_ppm": 250.0,
            "depth_err_ppm": 10.0,
            "warnings": [],
            "failure_reason": None,
            "variant_config": {"variant_id": "variant_a"},
            "gp_hyperparams": None,
            "gp_fit_diagnostics": None,
        }
    ]
    report_payload = {
        "summary": {
            "tic_id": 321,
            "input_depth_ppm": 225.0,
            "ephemeris": {
                "period_days": 9.0,
                "t0_btjd": 2500.25,
                "duration_hours": 3.0,
            },
            "sectors_used": [21, 22],
        }
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")

    seen: dict[str, Any] = {}

    def _fake_download_all_sectors(self, tic_id: int, flux_type: str, sectors: list[int] | None = None):
        seen["download"] = {"tic_id": tic_id, "flux_type": flux_type, "sectors": sectors}
        return [_FakeLightCurve()]

    monkeypatch.setattr("bittr_tess_vetter.cli.detrend_grid_cli.MASTClient.download_all_sectors", _fake_download_all_sectors)
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.detrend_grid_cli.compute_sensitivity_sweep_numpy",
        lambda **_kwargs: _FakeSweepResult(rows),
    )

    runner = CliRunner()
    result = runner.invoke(enrich_cli.cli, ["detrend-grid", "--report-file", str(report_path)])
    assert result.exit_code == 0, result.output
    payload = _json_from_cli_output(result.output)
    assert seen["download"] == {"tic_id": 321, "flux_type": "pdcsap", "sectors": [21, 22]}
    assert payload["provenance"]["inputs_source"] == "report_file"
    assert payload["provenance"]["sector_selection_source"] == "report_file"


def test_detrend_grid_best_variant_fallback_when_all_variants_fail(monkeypatch) -> None:
    rows = [
        {
            "variant_id": "variant_fail",
            "status": "failed",
            "backend": "numpy",
            "runtime_seconds": 0.01,
            "n_points_used": 200,
            "downsample_factor": 1,
            "outlier_policy": "none",
            "detrender": "transit_masked_bin_median",
            "score": None,
            "depth_hat_ppm": None,
            "depth_err_ppm": None,
            "warnings": [],
            "failure_reason": "mock failure",
            "variant_config": {
                "variant_id": "variant_fail",
                "detrender": "transit_masked_bin_median",
                "detrender_bin_hours": 6.0,
                "detrender_buffer_factor": 2.0,
                "detrender_sigma_clip": 5.0,
            },
            "gp_hyperparams": None,
            "gp_fit_diagnostics": None,
        }
    ]

    class _AllFailSweep:
        def to_dict(self) -> dict[str, Any]:
            return {
                "stable": False,
                "metric_variance": None,
                "score_spread_iqr_over_median": None,
                "depth_spread_iqr_over_median": None,
                "n_variants_total": 1,
                "n_variants_ok": 0,
                "n_variants_failed": 1,
                "best_variant_id": None,
                "worst_variant_id": None,
                "stability_threshold": 0.2,
                "notes": [],
                "sweep_table": rows,
            }

    monkeypatch.setattr("bittr_tess_vetter.cli.detrend_grid_cli.MASTClient", _FakeMASTClient)
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.detrend_grid_cli.compute_sensitivity_sweep_numpy",
        lambda **_kwargs: _AllFailSweep(),
    )

    runner = CliRunner()
    result = runner.invoke(enrich_cli.cli, _base_args())
    assert result.exit_code == 0, result.output
    payload = _json_from_cli_output(result.output)
    assert payload["verdict"] == "UNSTABLE"
    assert payload["result"]["verdict"] == "UNSTABLE"
    assert payload["best_variant"]["variant_id"] == "variant_fail"
    assert payload["best_variant"]["config"]["detrender"] == "transit_masked_bin_median"
    assert payload["recommended_detrend_flags"] == {
        "detrend": "transit_masked_bin_median",
        "bin_hours": 6.0,
        "buffer": 2.0,
        "sigma_clip": 5.0,
    }
    assert payload["result_summary"]["best_variant_id"] == "variant_fail"
    assert payload["result_summary"]["stable"] is False
    assert payload["result_summary"]["n_variants_total"] == 1
    assert payload["result_summary"]["depth_hat_ppm_range"] is None
