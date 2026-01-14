"""Cadence-level masking utilities for pixel-level algorithms.

The goal is to enforce consistent data hygiene across pixel modules:
- exclude known-bad cadences (quality != 0)
- exclude non-finite time stamps
- optionally require at least one finite pixel in the stamp
"""

from __future__ import annotations

from typing import Any

import numpy as np


def default_cadence_mask(
    *,
    time: np.ndarray[Any, np.dtype[np.floating[Any]]],
    flux: np.ndarray[Any, np.dtype[np.floating[Any]]],
    quality: np.ndarray[Any, np.dtype[np.integer[Any]]],
    require_finite_pixels: bool = True,
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    """Return a conservative cadence mask for pixel-level computations."""
    mask = (quality == 0) & np.isfinite(time)

    if require_finite_pixels:
        flat = flux.reshape(flux.shape[0], -1)
        mask &= np.any(np.isfinite(flat), axis=1)

    return mask


__all__ = ["default_cadence_mask"]
