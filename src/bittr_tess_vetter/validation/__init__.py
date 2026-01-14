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
from bittr_tess_vetter.validation.checks_catalog_wrapped import (
    ExoFOPTOILookupCheck,
    NearbyEBSearchCheck,
    register_catalog_checks,
)
from bittr_tess_vetter.validation.checks_lc_wrapped import (
    DepthStabilityCheck,
    DurationConsistencyCheck,
    OddEvenDepthCheck,
    SecondaryEclipseCheck,
    VShapeCheck,
    register_lc_checks,
)
from bittr_tess_vetter.validation.lc_checks import (
    check_depth_stability,
    check_duration_consistency,
    check_odd_even_depth,
    check_secondary_eclipse,
    check_v_shape,
)
from bittr_tess_vetter.validation.register_defaults import register_all_defaults
from bittr_tess_vetter.validation.registry import (
    DEFAULT_REGISTRY,
    CheckConfig,
    CheckInputs,
    CheckRegistry,
    CheckRequirements,
    CheckTier,
    VettingCheck,
    get_default_registry,
)
from bittr_tess_vetter.validation.result_schema import (
    CheckResult,
    CheckStatus,
    VettingBundleResult,
    error_result,
    ok_result,
    skipped_result,
)

__all__ = [
    # Result schema types
    "CheckResult",
    "CheckStatus",
    "VettingBundleResult",
    "error_result",
    "ok_result",
    "skipped_result",
    # Registry types
    "CheckConfig",
    "CheckInputs",
    "CheckRegistry",
    "CheckRequirements",
    "CheckTier",
    "VettingCheck",
    "DEFAULT_REGISTRY",
    "get_default_registry",
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
    # Function-based checks (V01-V05 LC-only)
    "check_depth_stability",
    "check_duration_consistency",
    "check_odd_even_depth",
    "check_secondary_eclipse",
    "check_v_shape",
    # VettingCheck wrapper classes (V01-V05 LC-only)
    "OddEvenDepthCheck",
    "SecondaryEclipseCheck",
    "DurationConsistencyCheck",
    "DepthStabilityCheck",
    "VShapeCheck",
    # VettingCheck wrapper classes (V06-V07 catalog)
    "NearbyEBSearchCheck",
    "ExoFOPTOILookupCheck",
    # Registration functions
    "register_lc_checks",
    "register_catalog_checks",
    "register_all_defaults",
]
