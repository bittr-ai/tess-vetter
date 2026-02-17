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
    assert "n_plausible_scenarios" in payload["physics_flags"]
    assert isinstance(payload["n_plausible_scenarios"], int)
    assert isinstance(payload["result"]["scenarios"], list)
    assert len(payload["result"]["scenarios"]) == 2
    assert isinstance(payload["result"]["physics_flags"], dict)
    assert isinstance(payload["result"]["n_plausible_scenarios"], int)
    assert payload["result"]["scenarios"] == payload["scenarios"]
    assert payload["result"]["physics_flags"] == payload["physics_flags"]
    assert payload["result"]["n_plausible_scenarios"] == payload["n_plausible_scenarios"]
    assert "verdict" in payload
    assert "verdict_source" in payload
    assert payload["result"]["verdict"] == payload["verdict"]
    assert payload["result"]["verdict_source"] == payload["verdict_source"]
    assert payload["result"]["reliability_summary"]["action_hint"] == payload["verdict"]
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
                "multiplicity_risk": {
                    "status": "ELEVATED",
                    "reasons": ["TARGET_RUWE_ELEVATED"],
                },
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
    assert payload["reliability_summary"]["multiplicity_risk"]["status"] == "ELEVATED"


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


def test_btv_dilution_report_file_inputs_override_candidate_flags(monkeypatch, tmp_path: Path) -> None:
    report_path = tmp_path / "dilution.report.json"
    report_path.write_text(
        json.dumps(
            {
                "report": {
                    "summary": {
                        "tic_id": 555,
                        "ephemeris": {
                            "period_days": 6.0,
                            "t0_btjd": 2450.0,
                            "duration_hours": 2.0,
                        },
                        "input_depth_ppm": 400.0,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    host_profile_path = tmp_path / "host_profile_report.json"
    host_profile_path.write_text(
        json.dumps(
            {
                "primary": {"tic_id": 555, "g_mag": 10.2, "radius_rsun": 1.0},
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

    monkeypatch.setattr(
        "bittr_tess_vetter.cli.dilution_cli._resolve_candidate_inputs",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("should not resolve TOI with report file")),
    )

    out_path = tmp_path / "dilution_report_file.json"
    runner = CliRunner()
    result = runner.invoke(
        dilution_command,
        [
            "--report-file",
            str(report_path),
            "--toi",
            "TOI-555.01",
            "--host-profile-file",
            str(host_profile_path),
            "--out",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Warning: --report-file provided; ignoring candidate input flags" in result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["inputs_summary"]["input_resolution"]["source"] == "report_file"
    assert payload["provenance"]["inputs_source"] == "report_file"
    assert payload["provenance"]["report_file"] == str(report_path.resolve())


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


def test_btv_dilution_autoresolves_primary_radius_when_missing_and_network_ok(monkeypatch, tmp_path: Path) -> None:
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
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.dilution_cli.load_auto_stellar_with_fallback",
        lambda **_kwargs: (
            {"radius": 0.93, "mass": 0.85, "tmag": 10.8},
            {"selected_source": "exofop_toi_table"},
        ),
    )

    host_profile_path = tmp_path / "host_profile_missing_radius.json"
    host_profile_path.write_text(
        json.dumps({"primary": {"tic_id": 123, "g_mag": 10.2}, "companions": []}),
        encoding="utf-8",
    )

    out_path = tmp_path / "dilution_missing_radius.json"
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
    assert payload["scenarios"][0]["host"]["radius_rsun"] == 0.93
    assert payload["provenance"]["primary_radius_resolution"]["attempted"] is True
    assert payload["provenance"]["primary_radius_resolution"]["resolved_from"] == "auto_stellar_fallback"
    assert payload["provenance"]["primary_radius_resolution"]["radius_rsun"] == 0.93


def test_btv_dilution_auto_radius_resolution_fail_open(monkeypatch, tmp_path: Path) -> None:
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

    def _raise_auto_stellar(**_kwargs: Any):
        raise RuntimeError("upstream stellar lookup timeout")

    monkeypatch.setattr(
        "bittr_tess_vetter.cli.dilution_cli.load_auto_stellar_with_fallback",
        _raise_auto_stellar,
    )

    host_profile_path = tmp_path / "host_profile_missing_radius_with_auto_error.json"
    host_profile_path.write_text(
        json.dumps({"primary": {"tic_id": 123, "g_mag": 10.2}, "companions": []}),
        encoding="utf-8",
    )

    out_path = tmp_path / "dilution_auto_error.json"
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
    assert payload["provenance"]["primary_radius_resolution"]["attempted"] is True
    assert payload["provenance"]["primary_radius_resolution"]["resolved_from"] is None
    assert payload["provenance"]["primary_radius_resolution"]["radius_rsun"] is None
    assert payload["provenance"]["primary_radius_resolution"]["error"] == "upstream stellar lookup timeout"
    assert payload["scenarios"][0]["scenario_plausibility"] == "unevaluated"


def test_btv_dilution_marks_implausible_depth_without_host_radius(tmp_path: Path) -> None:
    host_profile_path = tmp_path / "host_profile_depth_impossible.json"
    host_profile_path.write_text(
        json.dumps({"primary": {"tic_id": 123}, "companions": []}),
        encoding="utf-8",
    )

    out_path = tmp_path / "dilution_depth_impossible.json"
    runner = CliRunner()
    result = runner.invoke(
        dilution_command,
        [
            "--depth-ppm",
            "1200000",
            "--host-profile-file",
            str(host_profile_path),
            "--out",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    scenario = payload["scenarios"][0]
    assert scenario["true_depth_ppm"] > 1_000_000.0
    assert scenario["scenario_plausibility"] == "implausible_depth"
    assert scenario["physically_impossible"] is True
