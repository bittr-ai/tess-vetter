from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from click.testing import CliRunner

import tess_vetter.cli.enrich_cli as enrich_cli
from tess_vetter.cli import fpp_cli_v3


def _write_plan(path: Path) -> None:
    payload = {
        "schema_version": "cli.fpp.plan.v2",
        "created_at": "2026-02-26T00:00:00+00:00",
        "tic_id": 123,
        "period_days": 5.0,
        "t0_btjd": 2000.0,
        "duration_hours": 3.0,
        "depth_ppm_used": 500.0,
        "sectors_loaded": [10, 11],
        "cache_dir": str(path.parent),
        "runtime_artifacts": {"target_cached": True, "trilegal_cached": True},
        "inputs": {"depth_ppm_catalog": 500.0},
        "plan_signature": "sig",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_btv_help_lists_fpp_group_and_subcommands() -> None:
    runner = CliRunner()
    result = runner.invoke(enrich_cli.cli, ["fpp", "--help"])
    assert result.exit_code == 0
    assert "plan" in result.output
    assert "run" in result.output
    assert "sweep" in result.output
    assert "summary" in result.output
    assert "explain" in result.output


def test_removed_top_level_fpp_commands_fail_fast() -> None:
    runner = CliRunner()
    prep = runner.invoke(enrich_cli.cli, ["fpp-prepare"])
    run = runner.invoke(enrich_cli.cli, ["fpp-run"])
    assert prep.exit_code != 0
    assert run.exit_code != 0
    assert "removed" in prep.output.lower()
    assert "removed" in run.output.lower()


def test_fpp_run_enforces_same_tier_derived_knob_conflict(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path)

    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "fpp",
            "run",
            "--plan",
            str(plan_path),
            "--mode",
            "balanced",
            "--sampler-profile",
            "high",
            "--out",
            str(tmp_path / "out.json"),
        ],
    )
    assert result.exit_code != 0
    assert "Conflicting same-tier runtime inputs for sampler_profile" in result.output


def test_fpp_run_emits_v4_payload_with_policy_resolution(monkeypatch, tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    out_path = tmp_path / "run.json"
    _write_plan(plan_path)

    def _fake_calculate_fpp(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["overrides"]["mc_draws"] == 100000
        assert kwargs["overrides"]["max_points"] == 1500
        return {
            "fpp": 0.01,
            "nfpp": 0.0,
            "disposition": "VALIDATED",
            "base_seed": kwargs.get("seed", 7),
            "effective_config_hash": "abc123",
            "replicate_analysis": {"summary": {"requested_replicates": kwargs.get("replicates", 3)}},
        }

    monkeypatch.setattr("tess_vetter.cli.fpp_cli_v3.calculate_fpp", _fake_calculate_fpp)
    monkeypatch.setattr(
        "tess_vetter.cli.fpp_cli_v3.resolve_stellar_inputs",
        lambda **_kwargs: ({"radius": None, "mass": None, "tmag": None}, {"auto": None, "values": {}}),
    )

    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "fpp",
            "run",
            "--plan",
            str(plan_path),
            "--seed",
            "42",
            "--out",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "cli.fpp.v4"
    policy = payload["provenance"]["runtime"]["policy_resolution"]
    assert policy["effective_runtime_policy"]["mc_draws"] == 100000
    assert policy["effective_runtime_policy"]["max_points"] == 1500
    assert payload["provenance"]["runtime"]["timeout_policy"] == "auto_scaled"
    assert payload["provenance"]["runtime"]["timeout_seconds_requested"] is None
    assert float(payload["provenance"]["runtime"]["timeout_seconds_effective"]) >= 120.0
    assert payload["fpp_result"]["disposition"] == "VALIDATED"


def test_fpp_summary_and_explain_commands_read_result(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    summary_path = tmp_path / "summary.json"
    explain_path = tmp_path / "explain.json"

    payload = {
        "schema_version": "cli.fpp.v4",
        "fpp_result": {
            "fpp": 0.02,
            "disposition": "VALIDATED",
            "replicates": 3,
            "n_success": 3,
            "n_fail": 0,
            "effective_config_hash": "hash1",
            "replicate_analysis": {"summary": {"requested_replicates": 3}},
        },
        "provenance": {
            "runtime": {
                "scenario_id": "default",
                "policy_resolution": {"effective_runtime_policy": {"mode": "balanced"}},
                "degenerate_guard": {"attempts": [{"attempt": 1, "degenerate": False}]},
            }
        },
    }
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    runner = CliRunner()
    summary_res = runner.invoke(
        enrich_cli.cli,
        ["fpp", "summary", "--from", str(result_path), "--out", str(summary_path)],
    )
    assert summary_res.exit_code == 0, summary_res.output

    explain_res = runner.invoke(
        enrich_cli.cli,
        ["fpp", "explain", "--from", str(result_path), "--out", str(explain_path)],
    )
    assert explain_res.exit_code == 0, explain_res.output

    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    explain_payload = json.loads(explain_path.read_text(encoding="utf-8"))
    assert summary_payload["schema_version"] == "cli.fpp.summary.v1"
    assert explain_payload["schema_version"] == "cli.fpp.explain.v1"
    assert explain_payload["policy_resolution"]["effective_runtime_policy"]["mode"] == "balanced"


def test_fpp_sweep_writes_matrix_and_redundancy_artifacts(monkeypatch, tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    cfg_path = tmp_path / "sweep.yaml"
    out_dir = tmp_path / "sweep_out"
    _write_plan(plan_path)

    cfg_path.write_text(
        "\n".join(
            [
                "target:",
                "  toi: TOI-TEST.01",
                "base_runtime_policy:",
                "  mode: balanced",
                "matrix:",
                "  mc_draws: [100000, 100000]",
                "  max_points: [1500]",
                "seed_policy:",
                "  base_seed: 11",
                "  scenario_seed_strategy: scenario_index_offset",
                "  replicate_seed_strategy: replicate_index_offset",
                "  lock_seed_to_matrix: true",
                "execution:",
                "  parallelism: sequential",
                "  ordering: matrix",
                "  max_workers: 1",
                "outputs:",
                "  format: json",
            ]
        ),
        encoding="utf-8",
    )

    call_count = {"n": 0}

    seen_scenarios: list[str] = []

    def _fake_run_from_plan(**kwargs: Any) -> dict[str, Any]:
        call_count["n"] += 1
        seen_scenarios.append(str(kwargs.get("scenario_id")))
        # force same effective hash for both scenarios to trigger redundancy grouping
        return {
            "schema_version": "cli.fpp.v4",
            "fpp_result": {
                "fpp": 0.01 + 0.01 * call_count["n"],
                "disposition": "VALIDATED",
                "effective_config_hash": "dupe_hash",
            },
            "provenance": {
                "runtime": {
                    "policy_resolution": {"resolution_trace": []},
                    "degenerate_guard": {"attempts": [{"attempt": 1, "degenerate": False}]},
                }
            },
        }

    monkeypatch.setattr("tess_vetter.cli.fpp_cli_v3._run_from_plan", _fake_run_from_plan)

    runner = CliRunner()
    result = runner.invoke(
        enrich_cli.cli,
        [
            "fpp",
            "sweep",
            "--plan",
            str(plan_path),
            "--config",
            str(cfg_path),
            "--out-dir",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen_scenarios == ["scenario_001", "scenario_002"]

    assert (out_dir / "matrix_summary.json").exists()
    assert (out_dir / "matrix_ranked.csv").exists()
    assert (out_dir / "sweep_explain.json").exists()
    assert (out_dir / "matrix_redundancy.json").exists()

    redundancy = json.loads((out_dir / "matrix_redundancy.json").read_text(encoding="utf-8"))
    assert redundancy["groups"]
    assert redundancy["groups"][0]["effective_config_hash"] == "dupe_hash"


def test_execute_with_policy_retry_uses_single_total_timeout_budget(monkeypatch) -> None:
    monkeypatch.setattr(
        "tess_vetter.cli.fpp_cli_v3._is_degenerate_fpp_result",
        lambda payload: bool(payload.get("degenerate")),
    )
    resolved = fpp_cli_v3._ResolvedPolicy(
        requested_runtime_policy={},
        effective_runtime_policy={
            "mode": "balanced",
            "replicates": 3,
            "sampler_profile": "medium",
            "point_profile": "windowed",
            "fallback_policy": ["reduce_points_to_1000", "abort"],
            "mc_draws": 100000,
            "max_points": 1500,
        },
        resolution_trace=[],
        preset="standard",
        engine_overrides={"mc_draws": 100000, "max_points": 1500},
    )
    seen_timeouts: list[float] = []
    call_count = {"n": 0}

    def _run_attempt(_overrides: dict[str, Any], attempt_timeout: float | None) -> dict[str, Any]:
        call_count["n"] += 1
        seen_timeouts.append(float(attempt_timeout) if attempt_timeout is not None else 0.0)
        time.sleep(0.05)
        return {"degenerate": call_count["n"] == 1}

    _result, _attempts = fpp_cli_v3._execute_with_policy_retry(
        resolved=resolved,
        run_attempt=_run_attempt,
        total_timeout_seconds=1.0,
    )

    assert len(seen_timeouts) == 2
    assert seen_timeouts[1] < seen_timeouts[0]
