from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from click.testing import CliRunner

from bittr_tess_vetter.cli.followup_cli import followup_command


def test_btv_followup_success_contract_payload(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def _fake_resolve_candidate_inputs(**_kwargs: Any):
        return 123, 1.0, 1.0, 1.0, None, {
            "source": "toi_catalog",
            "resolved_from": "exofop_toi_table",
            "inputs": {
                "tic_id": 123,
            },
        }

    def _fake_resolve_followup_executor():
        def _executor(request: Any) -> dict[str, Any]:
            seen["request"] = request
            return {
                "files": [
                    {"filename": "spec_1.fits", "type": "Spectrum"},
                    {"filename": "img_1.png", "type": "Image"},
                ],
                "vetting_notes": ["First note", "Second note"],
                "summary": {
                    "n_files": 2,
                    "n_vetting_notes": 2,
                    "files_source": "cache_manifest",
                },
                "provenance_extra": {
                    "capabilities": {
                        "image_rendering": {"requested": True, "available": True, "used": True},
                        "spectra_content": {"mode": "raw", "raw_available": True},
                    },
                    "warnings": ["renderer:ok"],
                },
            }

        return _executor

    monkeypatch.setattr(
        "bittr_tess_vetter.cli.followup_cli._resolve_candidate_inputs",
        _fake_resolve_candidate_inputs,
    )
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.followup_cli._resolve_followup_executor",
        _fake_resolve_followup_executor,
    )

    notes_path = tmp_path / "notes.txt"
    notes_path.write_text("ignored by fake executor\n", encoding="utf-8")

    out_path = tmp_path / "followup.json"
    runner = CliRunner()
    result = runner.invoke(
        followup_command,
        [
            "TOI-123.01",
            "--network-ok",
            "--cache-dir",
            str(tmp_path),
            "--render-images",
            "--include-raw-spectra",
            "--max-files",
            "2",
            "--notes-file",
            str(notes_path),
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen["request"].tic_id == 123
    assert seen["request"].toi == "TOI-123.01"
    assert seen["request"].render_images is True
    assert seen["request"].include_raw_spectra is True
    assert seen["request"].max_files == 2

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "cli.followup.v1"
    assert payload["files"][0]["filename"] == "spec_1.fits"
    assert payload["vetting_notes"] == ["First note", "Second note"]
    assert payload["summary"]["n_files"] == 2
    assert payload["summary"]["n_vetting_notes"] == 2
    assert payload["result"]["files"] == payload["files"]
    assert payload["result"]["vetting_notes"] == payload["vetting_notes"]
    assert payload["result"]["summary"] == payload["summary"]
    assert payload["verdict"] == "FOLLOWUP_AVAILABLE"
    assert payload["verdict_source"] == "$.summary.n_files"
    assert payload["result"]["verdict"] == payload["verdict"]
    assert payload["result"]["verdict_source"] == payload["verdict_source"]
    assert payload["inputs_summary"]["tic_id"] == 123
    assert payload["inputs_summary"]["input_resolution"]["source"] == "toi_catalog"
    assert payload["provenance"]["inputs_source"] == "toi_catalog"
    assert payload["provenance"]["options"] == {
        "network_ok": True,
        "cache_dir": str(tmp_path),
        "render_images": True,
        "include_raw_spectra": True,
        "max_files": 2,
        "skip_notes": False,
        "notes_file": str(notes_path),
    }
    assert payload["provenance"]["capabilities"]["image_rendering"]["used"] is True
    assert payload["provenance"]["warnings"] == ["renderer:ok"]


def test_btv_followup_report_file_inputs_override_toi(monkeypatch, tmp_path: Path) -> None:
    report_path = tmp_path / "followup.report.json"
    report_path.write_text(
        json.dumps(
            {
                    "report": {
                        "summary": {
                            "tic_id": 555,
                            "toi": "TOI-555.01",
                            "ephemeris": {
                                "period_days": 6.0,
                                "t0_btjd": 2450.0,
                                "duration_hours": 2.0,
                        },
                    },
                    "provenance": {"sectors_used": [13, 14]},
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "bittr_tess_vetter.cli.followup_cli._resolve_candidate_inputs",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("should not resolve TOI with report file")),
    )
    monkeypatch.setattr(
        "bittr_tess_vetter.cli.followup_cli._resolve_followup_executor",
        lambda: (lambda _request: {"files": [], "vetting_notes": [], "summary": {}}),
    )

    out_path = tmp_path / "followup_report_file.json"
    runner = CliRunner()
    result = runner.invoke(
        followup_command,
        [
            "--report-file",
            str(report_path),
            "--toi",
            "TOI-555.01",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Warning: --report-file provided; ignoring --toi" in result.output

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["inputs_summary"]["tic_id"] == 555
    assert payload["inputs_summary"]["toi"] == "TOI-555.01"
    assert payload["inputs_summary"]["input_resolution"]["source"] == "report_file"
    assert payload["provenance"]["inputs_source"] == "report_file"
    assert payload["provenance"]["report_file"] == str(report_path.resolve())


def test_btv_followup_rejects_skip_notes_with_notes_file(tmp_path: Path) -> None:
    notes_path = tmp_path / "notes.txt"
    notes_path.write_text("line\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        followup_command,
        [
            "--tic-id",
            "123",
            "--skip-notes",
            "--notes-file",
            str(notes_path),
        ],
    )

    assert result.exit_code == 1
    assert "--skip-notes cannot be combined with --notes-file" in result.output
