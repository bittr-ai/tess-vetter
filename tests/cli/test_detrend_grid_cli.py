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
    payload = json.loads(result.output)

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
    assert "provenance" in payload
    assert "ranked_sweep_table" in payload
    assert payload["best_variant"]["variant_id"] == "variant_a"
    assert payload["provenance"]["command"] == "detrend-grid"


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

    payload_a = json.loads(result_a.output)
    payload_b = json.loads(result_b.output)
    payload_default = json.loads(result_default.output)
    payload_zero = json.loads(result_zero.output)

    assert payload_a["best_variant"]["variant_id"] == payload_b["best_variant"]["variant_id"]
    assert payload_a["ranked_sweep_table"][0]["variant_id"] == payload_b["ranked_sweep_table"][0]["variant_id"]
    assert payload_default["best_variant"]["variant_id"] == payload_zero["best_variant"]["variant_id"]
    assert payload_default["provenance"]["random_seed"] == 0
