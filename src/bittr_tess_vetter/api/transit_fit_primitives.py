"""Low-level physical transit fitting primitives for the public API.

This module exists for host applications that need stable access to the batman-
based fitting building blocks (optimize/MCMC, model evaluation) without
importing `bittr_tess_vetter.transit.*` internals directly.

Note: The higher-level, researcher-facing API is `bittr_tess_vetter.api.transit_fit`.
"""

from __future__ import annotations

from bittr_tess_vetter.transit.batman_model import (  # noqa: F401
    ParameterEstimate,
    TransitFitResult,
    compute_batman_model,
    compute_derived_parameters,
    compute_uncertainties,
    detect_exposure_time,
    fit_mcmc,
    fit_optimize,
    fit_transit_model,
    get_ld_coefficients,
    quick_estimate,
)

__all__ = [
    "ParameterEstimate",
    "TransitFitResult",
    "detect_exposure_time",
    "quick_estimate",
    "get_ld_coefficients",
    "compute_batman_model",
    "compute_derived_parameters",
    "fit_optimize",
    "fit_mcmc",
    "compute_uncertainties",
    "fit_transit_model",
]

