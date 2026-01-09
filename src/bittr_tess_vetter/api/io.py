"""Host-facing I/O helpers (network + caching).

This module is intentionally a thin API wrapper around `bittr_tess_vetter.io`.
Downstream applications should import from `bittr_tess_vetter.api.io` rather than
deep-importing implementation modules.
"""

from bittr_tess_vetter.io import (
    DEFAULT_QUALITY_MASK,
    QUALITY_FLAG_BITS,
    DownloadPhase,
    DownloadProgress,
    LightCurveNotFoundError,
    MASTClient,
    MASTClientError,
    NameResolutionError,
    PersistentCache,
    ProgressCallback,
    ResolvedTarget,
    SearchResult,
    SessionCache,
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

