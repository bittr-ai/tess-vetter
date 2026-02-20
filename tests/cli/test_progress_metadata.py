from __future__ import annotations

from pathlib import Path

import pytest

from tess_vetter.cli.progress_metadata import (
    ProgressIOError,
    build_single_candidate_progress,
    decide_resume_for_single_candidate,
    read_progress_metadata,
    write_progress_metadata_atomic,
)


def _candidate() -> dict[str, float | int]:
    return {
        "tic_id": 123456,
        "period_days": 12.5,
        "t0_btjd": 2000.25,
        "duration_hours": 3.2,
    }


def test_build_progress_completed_aligns_enrich_counters() -> None:
    payload = build_single_candidate_progress(
        command="vet",
        output_path="/tmp/out.json",
        candidate=_candidate(),
        resume=True,
        status="completed",
        wall_time_seconds=2.5,
    )

    assert payload["mode"] == "single_candidate"
    assert payload["schema_version"] == 1
    assert payload["total_input"] == 1
    assert payload["processed"] == 1
    assert payload["skipped_resume"] == 0
    assert payload["errors"] == 0
    assert payload["last_candidate_key"] == payload["candidate"]["candidate_key"]


def test_write_then_read_progress_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "candidate.progress.json"
    payload = build_single_candidate_progress(
        command="report",
        output_path=tmp_path / "report.json",
        candidate=_candidate(),
        resume=False,
        status="running",
    )

    write_progress_metadata_atomic(path, payload)
    loaded = read_progress_metadata(path)

    assert loaded is not None
    assert loaded["command"] == "report"
    assert loaded["status"] == "running"
    assert loaded["last_candidate_key"] == payload["last_candidate_key"]


def test_read_progress_missing_file_returns_none(tmp_path: Path) -> None:
    assert read_progress_metadata(tmp_path / "missing.progress.json") is None


def test_read_progress_invalid_schema_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.progress.json"
    path.write_text('{"schema_version": 999, "mode": "single_candidate", "command": "vet"}\n')

    with pytest.raises(ProgressIOError):
        read_progress_metadata(path)


def test_resume_decision_true_for_completed_candidate_with_output(tmp_path: Path) -> None:
    candidate = _candidate()
    progress = build_single_candidate_progress(
        command="vet",
        output_path=tmp_path / "out.json",
        candidate=candidate,
        resume=True,
        status="completed",
    )

    decision = decide_resume_for_single_candidate(
        command="vet",
        candidate=candidate,
        resume=True,
        output_exists=True,
        progress=progress,
    )

    assert decision == {"resume": True, "reason": "already_processed"}


def test_resume_decision_false_on_candidate_mismatch(tmp_path: Path) -> None:
    progress = build_single_candidate_progress(
        command="vet",
        output_path=tmp_path / "out.json",
        candidate=_candidate(),
        resume=True,
        status="completed",
    )
    other_candidate = dict(_candidate())
    other_candidate["t0_btjd"] = 2000.35

    decision = decide_resume_for_single_candidate(
        command="vet",
        candidate=other_candidate,
        resume=True,
        output_exists=True,
        progress=progress,
    )

    assert decision == {"resume": False, "reason": "candidate_mismatch"}


def test_progress_write_failure_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "candidate.progress.json"
    payload = build_single_candidate_progress(
        command="report",
        output_path=tmp_path / "report.json",
        candidate=_candidate(),
        resume=False,
        status="running",
    )

    def _raise_replace(_src: object, _dst: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("tess_vetter.cli.progress_metadata.os.replace", _raise_replace)

    with pytest.raises(ProgressIOError):
        write_progress_metadata_atomic(path, payload)
