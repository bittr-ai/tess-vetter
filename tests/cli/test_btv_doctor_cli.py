from __future__ import annotations

import json

from click.testing import CliRunner

from tess_vetter.cli.common_cli import EXIT_DATA_UNAVAILABLE
from tess_vetter.cli.doctor_cli import DoctorCheck, doctor_command


def test_doctor_json_ready_payload(monkeypatch) -> None:
    monkeypatch.setattr(
        "tess_vetter.cli.doctor_cli._run_doctor_checks",
        lambda profile: [
            DoctorCheck(
                name="console_script:btv",
                ok=True,
                detail="ok",
                remediation=None,
            )
        ],
    )
    runner = CliRunner()
    result = runner.invoke(doctor_command, ["--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ready"] is True
    assert payload["profile"] == "vet"


def test_doctor_missing_dependency_returns_data_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        "tess_vetter.cli.doctor_cli._run_doctor_checks",
        lambda profile: [
            DoctorCheck(
                name="module:lightkurve",
                ok=False,
                detail="missing",
                remediation="Install runtime dependency: pip install lightkurve",
            )
        ],
    )
    runner = CliRunner()
    result = runner.invoke(doctor_command, [])
    assert result.exit_code == EXIT_DATA_UNAVAILABLE
    assert "Environment is not ready" in result.output
    assert "pip install lightkurve" in result.output

