"""Transit compute primitives for the public API.

This module exists to give host applications a stable import path under
`tess_vetter.api.*` for core transit masking and simple depth
measurement utilities.

Delegates to `tess_vetter.compute.transit`.
"""

from __future__ import annotations

from tess_vetter.compute.transit import (  # noqa: F401
    MAX_SNR,
    detect_transit,
    fold_transit,
    get_transit_mask,
    measure_depth,
)

__all__ = [
    "MAX_SNR",
    "get_transit_mask",
    "measure_depth",
    "fold_transit",
    "detect_transit",
]
