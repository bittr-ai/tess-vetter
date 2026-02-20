from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from tess_vetter.cli.toi_query_cli import toi_query_command
from tess_vetter.platform.catalogs.exofop_toi_table import ExoFOPToiTable


def _fake_table() -> ExoFOPToiTable:
    return ExoFOPToiTable(
        fetched_at_unix=1234.5,
        headers=[
            "toi",
            "tic_id",
            "period",
            "snr",
            "tfopwg_disposition",
            "duration_hours",
        ],
        rows=[
            {
                "toi": "300.01",
                "tic_id": "111",
                "period": "5.2",
                "snr": "11.0",
                "tfopwg_disposition": "PC",
                "duration_hours": "2.4",
            },
            {
                "toi": "300.02",
                "tic_id": "112",
                "period": "4.8",
                "snr": "8.5",
                "tfopwg_disposition": "FP",
                "duration_hours": "2.2",
            },
            {
                "toi": "300.03",
                "tic_id": "113",
                "period": "bad",
                "snr": "15.0",
                "tfopwg_disposition": "PC",
                "duration_hours": "2.0",
            },
        ],
    )


def test_btv_toi_query_json_contract_includes_query_and_source_stats(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "tess_vetter.cli.toi_query_cli.fetch_exofop_toi_table",
        lambda **_: _fake_table(),
    )

    out_path = tmp_path / "toi_query.json"
    runner = CliRunner()
    result = runner.invoke(
        toi_query_command,
        [
            "--period-min",
            "5.0",
            "--disposition",
            "PC",
            "--exclude-false-positives",
            "--sort-by",
            "snr",
            "--sort-descending",
            "--max-results",
            "1",
            "--columns",
            "toi,snr,tfopwg_disposition",
            "--cache-ttl",
            "123",
            "--format",
            "json",
            "--output",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "cli.toi_query.v1"
    assert payload["query"]["numeric_ranges"]["period"] == {"min": 5.0, "max": None}
    assert payload["query"]["include_dispositions"] == ["PC"]
    assert payload["query"]["max_results"] == 1
    assert payload["source_stats"]["source_rows"] == 3
    assert payload["source_stats"]["matched_rows_before_limit"] == 1
    assert payload["source_stats"]["returned_rows"] == 1
    assert payload["source_stats"]["skipped_non_numeric_rows"] == 1
    assert payload["results"] == [{"toi": "300.01", "snr": "11.0", "tfopwg_disposition": "PC"}]


def test_btv_toi_query_csv_output_to_stdout(monkeypatch) -> None:
    monkeypatch.setattr(
        "tess_vetter.cli.toi_query_cli.fetch_exofop_toi_table",
        lambda **_: _fake_table(),
    )
    runner = CliRunner()
    result = runner.invoke(
        toi_query_command,
        [
            "--period-min",
            "4.0",
            "--format",
            "csv",
            "--sort-by",
            "period",
            "--columns",
            "toi,tic_id",
        ],
    )

    assert result.exit_code == 0, result.output
    lines = [line.strip() for line in result.output.strip().splitlines()]
    assert lines[0] == "toi,tic_id"
    assert lines[1] == "300.02,112"
    assert lines[2] == "300.01,111"
