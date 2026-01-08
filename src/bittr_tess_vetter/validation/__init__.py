"""Vetting checks and utilities (domain-only).

This package contains array-in/array-out computations for common vetting checks.
It intentionally excludes any platform-layer guardrail/validity frameworks and
any evidence/manifest infrastructure.

Use per-module imports for optional integrations (catalog/exovetter/triceratops).
"""

from __future__ import annotations

from bittr_tess_vetter.validation.base import (
    AggregationConfig,
    CheckConfig,
    CheckID,
    VetterCheck,
    VetterRegistry,
    VetterResult,
    aggregate_results,
    bin_phase_curve,
    compute_disposition,
    compute_verdict,
    count_transits,
    generate_summary,
    get_check,
    get_in_transit_mask,
    get_odd_even_transit_indices,
    get_out_of_transit_mask,
    get_registry,
    make_result,
    measure_transit_depth,
    phase_fold,
    register_check,
    search_secondary_eclipse,
    sigma_clip,
)

from bittr_tess_vetter.validation.checks_basic import (
    DepthCheck,
    DurationCheck,
    OddEvenCheck,
    SNRCheck,
    get_basic_checks,
    run_basic_checks,
)

from bittr_tess_vetter.validation.checks_pixel import (
    ApertureDependenceCheck,
    CentroidShiftCheck,
    PixelLevelLCCheck,
    PixelLevelLCResult,
    check_aperture_dependence_with_tpf,
    check_centroid_shift_with_tpf,
    check_pixel_level_lc_with_tpf,
    compute_pixel_level_depths,
    compute_pixel_level_lc_check,
)

from bittr_tess_vetter.validation.checks_secondary import (
    SECONDARY_CHECKS,
    CentroidCheck,
    ContaminationCheck,
    SecondaryEclipseCheck,
    run_secondary_checks,
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
    # Base framework
    "AggregationConfig",
    "CheckConfig",
    "CheckID",
    "VetterCheck",
    "VetterRegistry",
    "VetterResult",
    "aggregate_results",
    "bin_phase_curve",
    "compute_disposition",
    "compute_verdict",
    "count_transits",
    "generate_summary",
    "get_check",
    "get_in_transit_mask",
    "get_odd_even_transit_indices",
    "get_out_of_transit_mask",
    "get_registry",
    "make_result",
    "measure_transit_depth",
    "phase_fold",
    "register_check",
    "search_secondary_eclipse",
    "sigma_clip",
    # Basic checks
    "DepthCheck",
    "DurationCheck",
    "OddEvenCheck",
    "SNRCheck",
    "get_basic_checks",
    "run_basic_checks",
    # Pixel checks
    "ApertureDependenceCheck",
    "CentroidShiftCheck",
    "PixelLevelLCCheck",
    "PixelLevelLCResult",
    "check_aperture_dependence_with_tpf",
    "check_centroid_shift_with_tpf",
    "check_pixel_level_lc_with_tpf",
    "compute_pixel_level_depths",
    "compute_pixel_level_lc_check",
    # Secondary checks
    "SECONDARY_CHECKS",
    "CentroidCheck",
    "ContaminationCheck",
    "SecondaryEclipseCheck",
    "run_secondary_checks",
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

