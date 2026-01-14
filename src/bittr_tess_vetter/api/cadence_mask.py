"""Cadence masking utilities for the public API.

Delegates to `bittr_tess_vetter.pixel.cadence_mask`.
"""

from __future__ import annotations

from bittr_tess_vetter.pixel.cadence_mask import default_cadence_mask

__all__ = ["default_cadence_mask"]
