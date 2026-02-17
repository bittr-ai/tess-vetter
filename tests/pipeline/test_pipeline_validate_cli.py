from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from bittr_tess_vetter.cli.common_cli import EXIT_INPUT_ERROR
from bittr_tess_vetter.cli.enrich_cli import cli


def test_pipeline_validate_cli_valid_profile_human_summary() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["pipeline", "validate", "--profile", "triage_fast"])

    assert result.exit_code == 0, result.output
    assert "valid profile=triage_fast composition_id=triage_fast step_count=" in result.output


def test_pipeline_validate_and_profiles_include_robustness_composition() -> None:
    runner = CliRunner()

    profiles_result = runner.invoke(cli, ["pipeline", "profiles"])
    assert profiles_result.exit_code == 0, profiles_result.output
    profile_names = [line.strip() for line in profiles_result.output.splitlines() if line.strip()]
    assert "robustness_composition" in profile_names

    validate_result = runner.invoke(cli, ["pipeline", "validate", "--profile", "robustness_composition"])
    assert validate_result.exit_code == 0, validate_result.output
    assert (
        "valid profile=robustness_composition composition_id=robustness_composition step_count=8"
        in validate_result.output
    )


def test_pipeline_validate_cli_rejects_mutually_exclusive_profile_and_composition_file(tmp_path: Path) -> None:
    composition_file = tmp_path / "composition.json"
    composition_file.write_text('{"schema_version":"pipeline.composition.v1","id":"x","steps":[{"id":"s1","op":"report"}]}')

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "pipeline",
            "validate",
            "--profile",
            "triage_fast",
            "--composition-file",
            str(composition_file),
        ],
    )

    assert result.exit_code == EXIT_INPUT_ERROR
    assert "Provide exactly one of --profile or --composition-file." in result.output


def test_pipeline_validate_cli_invalid_schema_json_output(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad_composition.json"
    bad_file.write_text('{"schema_version":"pipeline.composition.v1","id":"bad","steps":[]}', encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["pipeline", "validate", "--composition-file", str(bad_file), "--json"],
    )

    assert result.exit_code == EXIT_INPUT_ERROR
    lines = [line for line in result.output.splitlines() if line.strip()]
    payload = json.loads(lines[0])
    assert payload["valid"] is False
    assert payload["profile"] is None
    assert payload["composition_id"] is None
    assert payload["step_count"] == 0
    assert payload["errors"]
