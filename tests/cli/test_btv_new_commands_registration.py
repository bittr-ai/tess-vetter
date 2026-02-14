from __future__ import annotations

from click.testing import CliRunner

from bittr_tess_vetter.cli.enrich_cli import cli


def test_top_level_help_lists_new_api_addition_commands() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0, result.output
    assert "localize" in result.output
    assert "fit" in result.output
    assert "periodogram" in result.output
