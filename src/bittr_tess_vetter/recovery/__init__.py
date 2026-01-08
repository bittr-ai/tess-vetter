"""Transit recovery for active stars.

This module provides tools for recovering transit signals from active stars
where stellar variability overwhelms the transit signal.

Exports:
- Primitives: estimate_rotation_period, remove_stellar_variability,
              detrend_for_recovery, stack_transits, fit_trapezoid, count_transits
- Results: StackedTransit, TrapezoidFit

Note: Transit masking is now done via TLS transit_mask() - see
transitleastsquares.transit_mask() for transit mask generation.
"""

from __future__ import annotations

from bittr_tess_vetter.recovery.primitives import (
    count_transits,
    detrend_for_recovery,
    estimate_rotation_period,
    fit_trapezoid,
    remove_stellar_variability,
    stack_transits,
)
from bittr_tess_vetter.recovery.result import StackedTransit, TrapezoidFit

__all__ = [
    # Primitives
    "estimate_rotation_period",
    "remove_stellar_variability",
    "detrend_for_recovery",
    "stack_transits",
    "fit_trapezoid",
    "count_transits",
    # Result types
    "StackedTransit",
    "TrapezoidFit",
]
