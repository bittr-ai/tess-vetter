"""Pixel host hypothesis scoring API surface (host-facing).

This module provides a stable facade for the lightweight PRF-lite host
hypothesis scoring types and thresholds.
"""

from __future__ import annotations

from bittr_tess_vetter.compute.pixel_host_hypotheses import (  # noqa: F401
    FLIP_RATE_FAIL_THRESHOLD,
    FLIP_RATE_WARN_THRESHOLD,
    MARGIN_RESOLVE_THRESHOLD,
    HypothesisScore,
)

__all__ = [
    "HypothesisScore",
    "MARGIN_RESOLVE_THRESHOLD",
    "FLIP_RATE_WARN_THRESHOLD",
    "FLIP_RATE_FAIL_THRESHOLD",
]

