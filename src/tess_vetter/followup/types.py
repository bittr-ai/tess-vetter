from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

FollowupStatusKind = Literal["ok", "skipped", "failed"]


@dataclass(frozen=True)
class FollowupFileClassification:
    filename: str
    exofop_type: str | None
    exofop_group: str
    extension: str
    format: str


@dataclass(frozen=True)
class RenderCapabilities:
    pdftoppm_path: str | None
    gs_path: str | None
    can_render_pdf: bool
    preferred_renderer: str | None


@dataclass(frozen=True)
class FollowupFileProcessingStatus:
    filename: str
    status: FollowupStatusKind
    reason: str | None = None
    details: dict[str, Any] | None = None


@dataclass(frozen=True)
class FitsHeaderExtractionResult:
    path: Path
    header: dict[str, Any]
    status: FollowupFileProcessingStatus
