"""I/O utilities (platform-facing)."""

from __future__ import annotations

from bittr_tess_vetter.platform.io.cache import PersistentCache, SessionCache
from bittr_tess_vetter.platform.io.mast_client import (
    DEFAULT_QUALITY_MASK,
    QUALITY_FLAG_BITS,
    DownloadPhase,
    DownloadProgress,
    LightCurveNotFoundError,
    MASTClient,
    MASTClientError,
    NameResolutionError,
    ProgressCallback,
    ResolvedTarget,
    SearchResult,
    TargetNotFoundError,
)

__all__ = [
    "MASTClient",
    "MASTClientError",
    "LightCurveNotFoundError",
    "TargetNotFoundError",
    "NameResolutionError",
    "ResolvedTarget",
    "SearchResult",
    "SessionCache",
    "PersistentCache",
    "QUALITY_FLAG_BITS",
    "DEFAULT_QUALITY_MASK",
    "DownloadProgress",
    "DownloadPhase",
    "ProgressCallback",
]
