"""Advanced building blocks for custom pipelines.

This module re-exports lower-level primitives that are supported but not
part of the "golden path" API. Use these when you need fine-grained control.

For most use cases, prefer the main `bittr_tess_vetter.api` exports.

Example:
    >>> from bittr_tess_vetter.api.primitives import (
    ...     fold, median_detrend, mask_transits, check_odd_even_depth
    ... )
    >>> # Build your own custom analysis pipeline
"""

from __future__ import annotations

# Re-export primitives catalog (for discovery)
from bittr_tess_vetter.api.primitives_catalog import (
    PRIMITIVES_CATALOG,
    PrimitiveInfo,
    list_primitives,
)

# Re-export compute primitives (pure-compute, no I/O)
from bittr_tess_vetter.compute.primitives import (
    AstroPrimitives,
    astro,
    box_model,
    fold,
    median_detrend,
    periodogram,
)

# Re-export pixel utilities
from bittr_tess_vetter.pixel.centroid import (
    CentroidResult,
    CentroidShiftConfig,
    TransitParams,
    compute_centroid_shift,
)

# Re-export catalog check functions
from bittr_tess_vetter.validation.checks_catalog import (
    run_exofop_toi_lookup,
    run_nearby_eb_search,
)

# Re-export individual check functions (legacy API, pre-VettingPipeline)
from bittr_tess_vetter.validation.lc_checks import (
    check_depth_stability,
    check_duration_consistency,
    check_odd_even_depth,
    check_secondary_eclipse,
    check_v_shape,
)

__all__ = [
    # Compute primitives (pure-compute, no I/O)
    "astro",
    "AstroPrimitives",
    "periodogram",
    "fold",
    "median_detrend",
    "box_model",
    # Legacy LC check functions (pre-VettingPipeline API)
    "check_odd_even_depth",
    "check_secondary_eclipse",
    "check_duration_consistency",
    "check_depth_stability",
    "check_v_shape",
    # Catalog check functions
    "run_nearby_eb_search",
    "run_exofop_toi_lookup",
    # Pixel utilities
    "compute_centroid_shift",
    "CentroidResult",
    "CentroidShiftConfig",
    "TransitParams",
    # Primitives catalog (discovery)
    "PRIMITIVES_CATALOG",
    "PrimitiveInfo",
    "list_primitives",
]
