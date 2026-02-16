from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from click.testing import CliRunner

import bittr_tess_vetter.cli.systematics_proxy_cli as systematics_proxy_cli
from bittr_tess_vetter.cli.systematics_proxy_cli import systematics_proxy_command


def test_btv_systematics_proxy_success_payload_contract(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def _fake_resolve_candidate_inputs(**_kwargs: Any):
        return 123, 7.25, 2450.1, 3.5, None, {
            "source": "toi_catalog",
            "resolved_from": "exofop_toi_table",
            "inputs": {
                "tic_id": 123,
                "period_days": 7.25,
                "t0_btjd": 2450.1,
                "duration_hours": 3.5,
                "depth_ppm": None,
            },
            "overrides": [],
            "errors": [],
        }

    def _fake_download_and_prepare_arrays(**_kwargs: Any):
        return (
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
            np.array([1.0, 0.999, 1.001], dtype=np.float64),
            np.array([True, False, True], dtype=bool),
            [14, 15],
        )

    class _FakeSystematicsProxyResult:
        def to_dict(self) -> dict[str, Any]:
            return {
                "score": 0.35,
                "few_point_top5_fraction": 0.72,
                "max_step_sigma": 10.1,
            }

    def _fake_compute_systematics_proxy(**kwargs: Any) -> _FakeSystematicsProxyResult:
        seen.update(kwargs)
        return _FakeSystematicsProxyResult()

    monkeypatch.setattr(
        "bittr_tess_vetter.cli.systematics_proxy_cli._resolve_candidate_inputs",
        _fake_resolve_candidate_inputs,
    )
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.systematics_proxy_cli._download_and_prepare_arrays",
        _fake_download_and_prepare_arrays,
    )
    monkeypatch.setattr(
        systematics_proxy_cli.systematics_api,
        "compute_systematics_proxy",
        _fake_compute_systematics_proxy,
    )

    out_path = tmp_path / "systematics_proxy.json"
    runner = CliRunner()
    result = runner.invoke(
        systematics_proxy_command,
        [
            "--toi",
            "123.01",
            "--network-ok",
            "--sectors",
            "14",
            "--sectors",
            "15",
            "--flux-type",
            "sap",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["period_days"] == 7.25
    assert seen["t0_btjd"] == 2450.1
    assert seen["duration_hours"] == 3.5
    assert np.array_equal(seen["valid_mask"], np.array([True, False, True], dtype=bool))

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "cli.systematics_proxy.v1"
    assert payload["systematics_proxy"]["score"] == 0.35
    assert payload["inputs_summary"]["input_resolution"]["source"] == "toi_catalog"
    assert payload["provenance"]["sectors_used"] == [14, 15]
    assert payload["provenance"]["options"]["flux_type"] == "sap"


def test_btv_systematics_proxy_missing_required_ephemeris_input_exits_1() -> None:
    runner = CliRunner()
    result = runner.invoke(
        systematics_proxy_command,
        [
            "--tic-id",
            "123",
        ],
    )
    assert result.exit_code == 1
    assert "Missing required inputs" in result.output


def test_btv_systematics_proxy_accepts_positional_toi_and_short_o(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def _fake_resolve_candidate_inputs(**kwargs: Any):
        seen.update(kwargs)
        return 123, 7.25, 2450.1, 3.5, None, {"source": "toi", "inputs": {"toi": kwargs.get("toi")}}

    monkeypatch.setattr(
        "bittr_tess_vetter.cli.systematics_proxy_cli._resolve_candidate_inputs",
        _fake_resolve_candidate_inputs,
    )
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.systematics_proxy_cli._download_and_prepare_arrays",
        lambda **_kwargs: (
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([1.0, 0.999], dtype=np.float64),
            np.array([True, True], dtype=bool),
            [14],
        ),
    )
    monkeypatch.setattr(
        systematics_proxy_cli.systematics_api,
        "compute_systematics_proxy",
        lambda **_kwargs: type("S", (), {"to_dict": lambda self: {"score": 0.1}})(),
    )

    out_path = tmp_path / "systematics_positional.json"
    runner = CliRunner()
    result = runner.invoke(systematics_proxy_command, ["TOI-5807.01", "-o", str(out_path)])
    assert result.exit_code == 0, result.output
    assert seen["toi"] == "TOI-5807.01"


def test_btv_systematics_proxy_rejects_mismatched_positional_and_option_toi() -> None:
    runner = CliRunner()
    result = runner.invoke(
        systematics_proxy_command,
        ["TOI-5807.01", "--toi", "TOI-4510.01"],
    )
    assert result.exit_code == 1
    assert "must match" in result.output
