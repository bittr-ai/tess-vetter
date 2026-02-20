"""Host-facing I/O helpers (network + caching).

This module is intentionally a thin API wrapper around `tess_vetter.platform.io`.
Downstream applications should import from `tess_vetter.api.io` rather than
deep-importing implementation modules.
"""

from tess_vetter.platform.io import (
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
