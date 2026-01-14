"""Vetting computations (metrics-only).

This package contains array-in/array-out computations for vetting checks.
It intentionally excludes any guardrail/policy aggregation (pass/warn/reject).
"""

from __future__ import annotations

from bittr_tess_vetter.validation.base import (
    bin_phase_curve,
    count_transits,
    get_in_transit_mask,
    get_odd_even_transit_indices,
    get_out_of_transit_mask,
    measure_transit_depth,
    phase_fold,
    search_secondary_eclipse,
    sigma_clip,
)
from bittr_tess_vetter.validation.lc_checks import (
    check_aperture_dependence,
    check_centroid_shift,
    check_depth_stability,
    check_duration_consistency,
    check_known_fp_match,
    check_nearby_eb_search,
    check_odd_even_depth,
    check_pixel_level_lc,
    check_secondary_eclipse,
    check_v_shape,
    run_all_checks,
)

__all__ = [
    # Base utilities
    "bin_phase_curve",
    "count_transits",
    "get_in_transit_mask",
    "get_odd_even_transit_indices",
    "get_out_of_transit_mask",
    "measure_transit_depth",
    "phase_fold",
    "search_secondary_eclipse",
    "sigma_clip",
    # Function-based checks
    "check_aperture_dependence",
    "check_centroid_shift",
    "check_depth_stability",
    "check_duration_consistency",
    "check_known_fp_match",
    "check_nearby_eb_search",
    "check_odd_even_depth",
    "check_pixel_level_lc",
    "check_secondary_eclipse",
    "check_v_shape",
    "run_all_checks",
]
