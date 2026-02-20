"""Cadence masking utilities for the public API.

Delegates to `tess_vetter.pixel.cadence_mask`.
"""

from __future__ import annotations

from tess_vetter.pixel.cadence_mask import default_cadence_mask

__all__ = ["default_cadence_mask"]
