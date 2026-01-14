"""Prefilter utilities for the public API.

Prefilters are lightweight checks/metrics used to quickly characterize a signal
before running heavier vetting steps.

Delegates to `bittr_tess_vetter.validation.prefilter`.
"""

from __future__ import annotations

from bittr_tess_vetter.validation.prefilter import (  # noqa: F401
    compute_depth_over_depth_err_snr,
    compute_phase_coverage,
)

__all__ = ["compute_depth_over_depth_err_snr", "compute_phase_coverage"]
