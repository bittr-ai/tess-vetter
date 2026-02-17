from __future__ import annotations

from click.testing import CliRunner

from bittr_tess_vetter.cli.enrich_cli import cli


def test_top_level_help_lists_new_api_addition_commands() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0, result.output
    assert "localize" in result.output
    assert "localize-host" in result.output
    assert "resolve-neighbors" in result.output
    assert "fit" in result.output
    assert "periodogram" in result.output
    assert "model-compete" in result.output
    assert "ephemeris-reliability" in result.output
    assert "activity" in result.output
    assert "fetch" in result.output
    assert "timing" in result.output
    assert "systematics-proxy" in result.output
    assert "dilution" in result.output
    assert "rv-feasibility" in result.output
    assert "toi-query" in result.output
    assert "contrast-curves" in result.output
    assert "contrast-curve-summary" in result.output
