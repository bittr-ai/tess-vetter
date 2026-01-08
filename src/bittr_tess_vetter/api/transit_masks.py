"""Transit window masks and simple depth primitives for the public API.

These are low-level utilities frequently needed by host apps to build
diagnostics. They are deliberately stable and deterministic.

Delegates to `bittr_tess_vetter.validation.base`.
"""

from __future__ import annotations

import numpy as np

from bittr_tess_vetter.validation.base import (  # noqa: F401
    count_transits,
    get_in_transit_mask,
    get_odd_even_transit_indices,
    get_out_of_transit_mask,
    measure_transit_depth,
)

__all__ = [
    "get_in_transit_mask",
    "get_out_of_transit_mask",
    "get_odd_even_transit_indices",
    "measure_transit_depth",
    "count_transits",
]

# Keep a small runtime assert for type sanity (numpy is required anyway)
_ = np.ndarray

