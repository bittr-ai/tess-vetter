"""Detrending API facade (host-facing).

The canonical implementations live in `tess_vetter.compute.detrend`, but
hosts should import from `tess_vetter.api.*` rather than deep-importing
from the compute package.
"""

from __future__ import annotations

from tess_vetter.compute.detrend import (  # noqa: F401
    WOTAN_AVAILABLE,
    bin_median_trend,
    flatten,
    flatten_with_wotan,
    median_detrend,
    normalize_flux,
    sigma_clip,
    wotan_flatten,
)
