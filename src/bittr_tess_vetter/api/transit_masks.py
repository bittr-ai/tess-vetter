"""Transit window masks and simple depth primitives for the public API.

These are low-level utilities frequently needed by host apps to build
diagnostics. They are deliberately stable and deterministic.

Delegates to `bittr_tess_vetter.validation.base`.
"""

from __future__ import annotations

import numpy as np

from bittr_tess_vetter.pixel.aperture_family import (
    _compute_out_of_transit_mask as _compute_out_of_transit_mask_windowed,
)
from bittr_tess_vetter.validation.base import (  # noqa: F401
    count_transits,
    get_in_transit_mask,
    get_odd_even_transit_indices,
    get_out_of_transit_mask,
    measure_transit_depth,
)


def get_out_of_transit_mask_windowed(
    time: np.ndarray,
    period: float,
    t0: float,
    duration_hours: float,
    *,
    oot_margin_mult: float = 1.5,
    oot_window_mult: float | None = 10.0,
) -> np.ndarray:
    """Out-of-transit mask with a local window around each transit.

    This is primarily useful for pixel-level work where a global baseline can
    be biased by long-term systematics. When `oot_window_mult` is None, it
    falls back to a global out-of-transit baseline excluding only the margin
    around transit.

    Delegates to `bittr_tess_vetter.pixel.aperture_family._compute_out_of_transit_mask`.
    """
    duration_days = float(duration_hours) / 24.0
    return _compute_out_of_transit_mask_windowed(
        np.asarray(time, dtype=np.float64),
        float(period),
        float(t0),
        duration_days,
        oot_margin_mult=float(oot_margin_mult),
        oot_window_mult=oot_window_mult,
    )


__all__ = [
    "get_in_transit_mask",
    "get_out_of_transit_mask",
    "get_out_of_transit_mask_windowed",
    "get_odd_even_transit_indices",
    "measure_transit_depth",
    "count_transits",
]

# Keep a small runtime assert for type sanity (numpy is required anyway)
_ = np.ndarray

