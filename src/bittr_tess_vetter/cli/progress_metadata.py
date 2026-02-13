"""CLI progress metadata helpers for single-candidate vet/report flows."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, TypedDict

from bittr_tess_vetter.pipeline import make_candidate_key

SingleCandidateCommand = Literal["vet", "report"]
SingleCandidateStatus = Literal["running", "completed", "error", "skipped_resume"]


class ProgressIOError(RuntimeError):
    """Raised when progress metadata cannot be read or written safely."""


class ProgressCandidate(TypedDict):
    tic_id: int
    period_days: float
    t0_btjd: float
    duration_hours: float
    candidate_key: str


class SingleCandidateProgress(TypedDict, total=False):
    schema_version: int
    mode: str
    command: SingleCandidateCommand
    output_path: str
    resume: bool
    total_input: int
    processed: int
    skipped_resume: int
    errors: int
    wall_time_seconds: float
    error_class_counts: dict[str, int]
    last_candidate_key: str
    updated_unix: float
    status: SingleCandidateStatus
    error_message: str
    candidate: ProgressCandidate


class ResumeDecision(TypedDict):
    resume: bool
    reason: str


def build_single_candidate_progress(
    *,
    command: SingleCandidateCommand,
    output_path: str | Path,
    candidate: Mapping[str, Any],
    resume: bool,
    status: SingleCandidateStatus,
    wall_time_seconds: float = 0.0,
    errors: int = 0,
    error_class: str | None = None,
    error_message: str | None = None,
) -> SingleCandidateProgress:
    """Build progress payload with counters compatible with enrich_worklist."""
    tic_id = int(candidate["tic_id"])
    period_days = float(candidate["period_days"])
    t0_btjd = float(candidate["t0_btjd"])
    duration_hours = float(candidate["duration_hours"])
    candidate_key = make_candidate_key(tic_id, period_days, t0_btjd)

    processed = 1 if status == "completed" else 0
    skipped_resume = 1 if status == "skipped_resume" else 0
    errors_count = int(errors)
    if status == "error" and errors_count <= 0:
        errors_count = 1

    error_class_counts: dict[str, int] = {}
    if errors_count > 0:
        error_class_counts[error_class or "Error"] = errors_count

    payload: SingleCandidateProgress = {
        "schema_version": 1,
        "mode": "single_candidate",
        "command": command,
        "output_path": str(output_path),
        "resume": bool(resume),
        "total_input": 1,
        "processed": processed,
        "skipped_resume": skipped_resume,
        "errors": errors_count,
        "wall_time_seconds": float(wall_time_seconds),
        "error_class_counts": error_class_counts,
        "last_candidate_key": candidate_key,
        "updated_unix": time.time(),
        "status": status,
        "candidate": {
            "tic_id": tic_id,
            "period_days": period_days,
            "t0_btjd": t0_btjd,
            "duration_hours": duration_hours,
            "candidate_key": candidate_key,
        },
    }
    if error_message:
        payload["error_message"] = str(error_message)
    return payload


def read_progress_metadata(path: str | Path) -> SingleCandidateProgress | None:
    """Read single-candidate progress metadata, returning None if missing."""
    progress_path = Path(path)
    if not progress_path.exists():
        return None

    try:
        payload = json.loads(progress_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ProgressIOError(f"Failed to read progress metadata: {progress_path}") from exc

    if not isinstance(payload, dict):
        raise ProgressIOError("Progress metadata must be a JSON object")
    if payload.get("schema_version") != 1:
        raise ProgressIOError("Unsupported progress metadata schema_version")
    if payload.get("mode") != "single_candidate":
        raise ProgressIOError("Progress metadata mode must be 'single_candidate'")
    if payload.get("command") not in ("vet", "report"):
        raise ProgressIOError("Progress metadata command must be 'vet' or 'report'")

    return payload  # type: ignore[return-value]


def write_progress_metadata_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Atomically persist progress metadata as JSON."""
    progress_path = Path(path)
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path: Path | None = None
    try:
        fd, raw_tmp = tempfile.mkstemp(
            dir=progress_path.parent,
            prefix=f".{progress_path.name}.",
            suffix=".tmp",
        )
        tmp_path = Path(raw_tmp)
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_handle:
            tmp_handle.write(json.dumps(dict(payload), indent=2, sort_keys=True))
            tmp_handle.write("\n")
            tmp_handle.flush()
            os.fsync(tmp_handle.fileno())
        os.replace(tmp_path, progress_path)
    except Exception as exc:
        raise ProgressIOError(f"Failed to write progress metadata: {progress_path}") from exc
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def decide_resume_for_single_candidate(
    *,
    command: SingleCandidateCommand,
    candidate: Mapping[str, Any],
    resume: bool,
    output_exists: bool,
    progress: SingleCandidateProgress | None,
) -> ResumeDecision:
    """Return whether a single-candidate command should skip due to resume state."""
    if not resume:
        return {"resume": False, "reason": "resume_disabled"}
    if progress is None:
        return {"resume": False, "reason": "missing_progress"}
    if not output_exists:
        return {"resume": False, "reason": "missing_output"}
    if progress.get("command") != command:
        return {"resume": False, "reason": "command_mismatch"}

    candidate_key = make_candidate_key(
        int(candidate["tic_id"]),
        float(candidate["period_days"]),
        float(candidate["t0_btjd"]),
    )
    if progress.get("last_candidate_key") != candidate_key:
        return {"resume": False, "reason": "candidate_mismatch"}

    status = progress.get("status")
    if status not in ("completed", "skipped_resume"):
        return {"resume": False, "reason": "incomplete_status"}
    if int(progress.get("errors", 0)) > 0:
        return {"resume": False, "reason": "has_errors"}

    if status == "completed" and int(progress.get("processed", 0)) >= 1:
        return {"resume": True, "reason": "already_processed"}
    if status == "skipped_resume" and int(progress.get("skipped_resume", 0)) >= 1:
        return {"resume": True, "reason": "already_skipped_resume"}
    return {"resume": False, "reason": "counter_mismatch"}
