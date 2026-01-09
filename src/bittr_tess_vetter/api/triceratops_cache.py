"""TRICERATOPS cache + helper API facade (host-facing).

This module exposes a small, reusable surface area around TRICERATOPS caching
and helper routines that host applications may need for:
- warming TRILEGAL downloads (slow external dependency)
- persisting TRICERATOPS Target objects between runs
- estimating durations when duration_hours is missing

The underlying implementation lives in `bittr_tess_vetter.validation.triceratops_fpp`,
but hosts should import from `bittr_tess_vetter.api.*` only.
"""

from __future__ import annotations

from bittr_tess_vetter.validation.triceratops_fpp import (  # noqa: F401
    CalculateFppInput,
    FppResult,
    _estimate_transit_duration as estimate_transit_duration,
    _get_disposition as get_disposition,
    _load_cached_triceratops_target as load_cached_triceratops_target,
    _prefetch_trilegal_csv as prefetch_trilegal_csv,
    _save_cached_triceratops_target as save_cached_triceratops_target,
)

# Back-compat aliases (explicitly exported so astro-arc-tess can keep stable names).
_estimate_transit_duration = estimate_transit_duration
_get_disposition = get_disposition
_load_cached_triceratops_target = load_cached_triceratops_target
_prefetch_trilegal_csv = prefetch_trilegal_csv
_save_cached_triceratops_target = save_cached_triceratops_target

