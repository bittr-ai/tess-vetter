"""Generic evidence envelope + provenance contracts for host applications.

These types are intentionally domain-agnostic:
- They do not encode specific vetting taxonomies or guardrail IDs.
- They provide a stable, JSON-serializable "envelope" for shipping computed
  evidence artifacts with provenance.

Host apps (like `astro-arc-tess`) can layer domain-specific meaning on top of
`evidence_type`, `subject`, and `guardrails_triggered`.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _get_git_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()[:12]
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


def compute_code_hash(module_paths: list[str] | None = None) -> str:
    """Compute a best-effort code identifier for provenance."""
    if module_paths is None:
        git_hash = _get_git_hash()
        if git_hash:
            return f"git:{git_hash}"
        return f"ts:{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"

    hasher = hashlib.sha256()
    for path in sorted(module_paths):
        try:
            with open(path, "rb") as f:
                hasher.update(f.read())
        except FileNotFoundError:
            hasher.update(path.encode())
    return hasher.hexdigest()[:16]


@dataclass(frozen=True)
class EvidenceProvenance:
    """Provenance for an evidence product."""

    dataset_id: str
    code_hash: str
    calibration_version: str
    parameters: dict[str, Any]
    created_date: str

    def __post_init__(self) -> None:
        if not self.dataset_id:
            raise ValueError("dataset_id cannot be empty")
        if not self.code_hash:
            raise ValueError("code_hash cannot be empty")
        if not self.calibration_version:
            raise ValueError("calibration_version cannot be empty")
        if not self.created_date:
            raise ValueError("created_date cannot be empty")

    @classmethod
    def create_now(
        cls,
        *,
        dataset_id: str,
        calibration_version: str,
        parameters: dict[str, Any] | None = None,
        code_hash: str | None = None,
    ) -> EvidenceProvenance:
        return cls(
            dataset_id=dataset_id,
            code_hash=code_hash or compute_code_hash(),
            calibration_version=calibration_version,
            parameters=parameters or {},
            created_date=datetime.now(UTC).isoformat(),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceProvenance:
        return cls(**data)


@dataclass
class EvidenceEnvelope:
    """A JSON-friendly envelope for an evidence product."""

    evidence_type: str
    subject: dict[str, Any]
    provenance: EvidenceProvenance
    summary: dict[str, Any]

    candidate_id: str | None = None
    blob_refs: dict[str, str] = field(default_factory=dict)
    guardrails_triggered: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.evidence_type:
            raise ValueError("evidence_type cannot be empty")
        if not isinstance(self.subject, dict) or not self.subject:
            raise ValueError("subject must be a non-empty dict")

    def add_guardrail(self, guardrail: str) -> None:
        if guardrail and guardrail not in self.guardrails_triggered:
            self.guardrails_triggered.append(guardrail)

    def add_warning(self, warning: str) -> None:
        if warning and warning not in self.warnings:
            self.warnings.append(warning)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_type": self.evidence_type,
            "subject": self.subject,
            "candidate_id": self.candidate_id,
            "provenance": self.provenance.to_dict(),
            "summary": self.summary,
            "blob_refs": self.blob_refs,
            "guardrails_triggered": self.guardrails_triggered,
            "warnings": self.warnings,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceEnvelope:
        provenance = EvidenceProvenance.from_dict(data["provenance"])
        return cls(
            evidence_type=data["evidence_type"],
            subject=data["subject"],
            candidate_id=data.get("candidate_id"),
            provenance=provenance,
            summary=data.get("summary", {}),
            blob_refs=data.get("blob_refs", {}),
            guardrails_triggered=data.get("guardrails_triggered", []),
            warnings=data.get("warnings", []),
        )


def save_evidence(envelope: EvidenceEnvelope, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(envelope.to_dict(), f, indent=2)
        f.write("\n")


def load_evidence(path: str | Path) -> EvidenceEnvelope:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("evidence JSON must be an object")
    return EvidenceEnvelope.from_dict(data)


__all__ = [
    "EvidenceEnvelope",
    "EvidenceProvenance",
    "compute_code_hash",
    "load_evidence",
    "save_evidence",
]

