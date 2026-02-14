from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from click.testing import CliRunner

from bittr_tess_vetter.cli.dilution_cli import dilution_command


def test_btv_dilution_success_payload_contract(monkeypatch, tmp_path: Path) -> None:
    def _fake_resolve_candidate_inputs(**_kwargs: Any):
        return 123, 8.2, 2100.5, 3.1, 420.0, {
            "source": "toi_catalog",
            "resolved_from": "exofop_toi_table",
            "inputs": {"tic_id": 123, "depth_ppm": 420.0},
        }

    monkeypatch.setattr(
        "bittr_tess_vetter.cli.dilution_cli._resolve_candidate_inputs",
        _fake_resolve_candidate_inputs,
    )

    host_profile_path = tmp_path / "host_profile.json"
    host_profile_path.write_text(
        json.dumps(
            {
                "primary": {"tic_id": 123, "g_mag": 10.2, "radius_rsun": 1.0},
                "companions": [
                    {
                        "source_id": 987654321,
                        "separation_arcsec": 8.0,
                        "g_mag": 11.0,
                        "radius_rsun": 0.5,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    out_path = tmp_path / "dilution.json"
    runner = CliRunner()
    result = runner.invoke(
        dilution_command,
        [
            "--toi",
            "123.01",
            "--network-ok",
            "--host-profile-file",
            str(host_profile_path),
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "cli.dilution.v1"
    assert isinstance(payload["scenarios"], list)
    assert len(payload["scenarios"]) == 2
    assert isinstance(payload["physics_flags"], dict)
    assert payload["inputs_summary"]["input_resolution"]["source"] == "toi_catalog"
    assert payload["provenance"]["host_profile_path"] == str(host_profile_path)
    assert payload["provenance"]["host_ambiguous"] is True


def test_btv_dilution_missing_host_profile_file_exits_1() -> None:
    runner = CliRunner()
    result = runner.invoke(
        dilution_command,
        [
            "--depth-ppm",
            "250",
        ],
    )
    assert result.exit_code == 1
    assert "Missing required option: --host-profile-file" in result.output
