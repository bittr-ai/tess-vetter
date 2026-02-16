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
    assert "Provide at least one: --host-profile-file or --reference-sources-file" in result.output


def test_btv_dilution_reference_sources_only_success_contract(monkeypatch, tmp_path: Path) -> None:
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

    reference_sources_path = tmp_path / "reference_sources.json"
    reference_sources_path.write_text(
        json.dumps(
            {
                "schema_version": "reference_sources.v1",
                "reference_sources": [
                    {
                        "source_id": "tic:123",
                        "role": "target",
                        "g_mag": 10.2,
                        "radius_rsun": 1.0,
                    },
                    {
                        "source_id": "987654321",
                        "separation_arcsec": 8.0,
                        "g_mag": 11.0,
                        "radius_rsun": 0.5,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    out_path = tmp_path / "dilution_reference_only.json"
    runner = CliRunner()
    result = runner.invoke(
        dilution_command,
        [
            "--toi",
            "123.01",
            "--network-ok",
            "--reference-sources-file",
            str(reference_sources_path),
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "cli.dilution.v1"
    assert len(payload["scenarios"]) == 2
    assert payload["provenance"]["host_profile_path"] is None
    assert payload["provenance"]["reference_sources_path"] == str(reference_sources_path)
    assert payload["provenance"]["host_ambiguous"] is True


def test_btv_dilution_partial_host_profile_supplemented_by_reference_sources(
    monkeypatch, tmp_path: Path
) -> None:
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

    host_profile_path = tmp_path / "host_profile_partial.json"
    host_profile_path.write_text(
        json.dumps(
            {
                "primary": {"tic_id": 123},
                "companions": [],
            }
        ),
        encoding="utf-8",
    )

    reference_sources_path = tmp_path / "reference_sources_supplement.json"
    reference_sources_path.write_text(
        json.dumps(
            {
                "schema_version": "reference_sources.v1",
                "reference_sources": [
                    {
                        "source_id": "tic:123",
                        "role": "target",
                        "g_mag": 10.2,
                        "radius_rsun": 1.0,
                    },
                    {
                        "source_id": "987654321",
                        "separation_arcsec": 8.0,
                        "g_mag": 11.0,
                        "radius_rsun": 0.5,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    out_path = tmp_path / "dilution_partial_profile.json"
    runner = CliRunner()
    result = runner.invoke(
        dilution_command,
        [
            "--toi",
            "123.01",
            "--network-ok",
            "--host-profile-file",
            str(host_profile_path),
            "--reference-sources-file",
            str(reference_sources_path),
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "cli.dilution.v1"
    assert len(payload["scenarios"]) == 2
    assert payload["provenance"]["host_profile_path"] == str(host_profile_path)
    assert payload["provenance"]["reference_sources_path"] == str(reference_sources_path)
    assert payload["provenance"]["host_ambiguous"] is True


def test_btv_dilution_reference_sources_bad_schema_exits_1(tmp_path: Path) -> None:
    reference_sources_path = tmp_path / "reference_sources_bad_schema.json"
    reference_sources_path.write_text(
        json.dumps(
            {
                "schema_version": "reference_sources.v0",
                "reference_sources": [],
            }
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        dilution_command,
        [
            "--depth-ppm",
            "250",
            "--reference-sources-file",
            str(reference_sources_path),
        ],
    )
    assert result.exit_code == 1
    assert "reference sources file schema_version must be 'reference_sources.v1'" in result.output


def test_btv_dilution_reference_companion_missing_separation_exits_1(tmp_path: Path) -> None:
    reference_sources_path = tmp_path / "reference_sources_missing_separation.json"
    reference_sources_path.write_text(
        json.dumps(
            {
                "schema_version": "reference_sources.v1",
                "reference_sources": [
                    {
                        "tic_id": 123,
                        "role": "target",
                        "g_mag": 10.2,
                    },
                    {
                        "source_id": 987654321,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        dilution_command,
        [
            "--depth-ppm",
            "250",
            "--reference-sources-file",
            str(reference_sources_path),
        ],
    )
    assert result.exit_code == 1
    assert "reference_sources[1].separation_arcsec is required for companions" in result.output


def test_btv_dilution_reference_sources_missing_tic_and_no_cli_tic_exits_1(tmp_path: Path) -> None:
    reference_sources_path = tmp_path / "reference_sources_missing_tic.json"
    reference_sources_path.write_text(
        json.dumps(
            {
                "schema_version": "reference_sources.v1",
                "reference_sources": [
                    {
                        "source_id": 987654321,
                        "separation_arcsec": 8.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        dilution_command,
        [
            "--depth-ppm",
            "250",
            "--reference-sources-file",
            str(reference_sources_path),
        ],
    )
    assert result.exit_code == 1
    assert "Unable to resolve host TIC ID" in result.output


def test_btv_dilution_accepts_resolve_neighbors_style_source_ids(monkeypatch, tmp_path: Path) -> None:
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

    reference_sources_path = tmp_path / "reference_sources_resolve_shape.json"
    reference_sources_path.write_text(
        json.dumps(
            {
                "schema_version": "reference_sources.v1",
                "reference_sources": [
                    {
                        "source_id": "tic:123",
                        "role": "target",
                        "g_mag": 10.2,
                    },
                    {
                        "source_id": "gaia:987654321",
                        "role": "companion",
                        "meta": {"separation_arcsec": 8.0, "phot_g_mean_mag": 11.0},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    out_path = tmp_path / "dilution_resolve_shape.json"
    runner = CliRunner()
    result = runner.invoke(
        dilution_command,
        [
            "--toi",
            "123.01",
            "--network-ok",
            "--reference-sources-file",
            str(reference_sources_path),
            "--out",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "cli.dilution.v1"
    assert len(payload["scenarios"]) == 2


def test_btv_dilution_accepts_positional_toi_and_short_o(monkeypatch, tmp_path: Path) -> None:
    def _fake_resolve_candidate_inputs(**kwargs: Any):
        return 123, 8.2, 2100.5, 3.1, 420.0, {
            "source": "toi_catalog",
            "resolved_from": "exofop_toi_table",
            "inputs": {"tic_id": 123, "depth_ppm": 420.0, "toi": kwargs.get("toi")},
        }

    monkeypatch.setattr(
        "bittr_tess_vetter.cli.dilution_cli._resolve_candidate_inputs",
        _fake_resolve_candidate_inputs,
    )

    host_profile_path = tmp_path / "host_profile_positional.json"
    host_profile_path.write_text(
        json.dumps({"primary": {"tic_id": 123, "g_mag": 10.2, "radius_rsun": 1.0}, "companions": []}),
        encoding="utf-8",
    )

    out_path = tmp_path / "dilution_positional.json"
    runner = CliRunner()
    result = runner.invoke(
        dilution_command,
        [
            "TOI-5807.01",
            "--network-ok",
            "--host-profile-file",
            str(host_profile_path),
            "-o",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "cli.dilution.v1"


def test_btv_dilution_rejects_mismatched_positional_and_option_toi(tmp_path: Path) -> None:
    host_profile_path = tmp_path / "host_profile_mismatch.json"
    host_profile_path.write_text(
        json.dumps({"primary": {"tic_id": 123, "g_mag": 10.2, "radius_rsun": 1.0}, "companions": []}),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        dilution_command,
        [
            "TOI-5807.01",
            "--toi",
            "TOI-4510.01",
            "--host-profile-file",
            str(host_profile_path),
        ],
    )
    assert result.exit_code == 1
    assert "must match" in result.output
