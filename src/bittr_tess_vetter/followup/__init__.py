from __future__ import annotations

from bittr_tess_vetter.followup.processing import (
    classify_followup_file,
    detect_render_capabilities,
    extract_fits_header,
)
from bittr_tess_vetter.followup.service import run_followup
from bittr_tess_vetter.followup.types import (
    FitsHeaderExtractionResult,
    FollowupFileClassification,
    FollowupFileProcessingStatus,
    FollowupStatusKind,
    RenderCapabilities,
)

__all__ = [
    "FitsHeaderExtractionResult",
    "FollowupFileClassification",
    "FollowupFileProcessingStatus",
    "FollowupStatusKind",
    "RenderCapabilities",
    "classify_followup_file",
    "detect_render_capabilities",
    "extract_fits_header",
    "run_followup",
]
