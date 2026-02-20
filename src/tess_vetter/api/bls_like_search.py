"""BLS-like search API surface (host-facing).

This module provides a stable facade for the lightweight BLS-like search used
by MLX discovery tools. Host applications should import from `api.*` rather
than `compute.*`.
"""

from __future__ import annotations

from tess_vetter.compute.bls_like_search import (  # noqa: F401
    BlsLikeCandidate,
    BlsLikeSearchResult,
    bls_like_search_numpy,
    bls_like_search_numpy_top_k,
)

__all__ = [
    "BlsLikeCandidate",
    "BlsLikeSearchResult",
    "bls_like_search_numpy",
    "bls_like_search_numpy_top_k",
]
