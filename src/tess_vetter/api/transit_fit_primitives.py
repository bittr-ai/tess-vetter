"""Low-level physical transit fitting primitives for the public API.

This module exists for host applications that need stable access to the batman-
based fitting building blocks (optimize/MCMC, model evaluation) without
importing `tess_vetter.transit.*` internals directly.

Note: The higher-level, researcher-facing API is `tess_vetter.api.transit_fit`.
"""

from __future__ import annotations

from tess_vetter.api.references import (
    BYRD_1995_LBFGSB,
    CLARET_2018,
    FOREMAN_MACKEY_2013,
    GOODMAN_WEARE_2010,
    KREIDBERG_2015,
    MANDEL_AGOL_2002,
    PARVIAINEN_2015,
    SEAGER_MALLEN_ORNELAS_2003,
    cite,
    cites,
)
from tess_vetter.transit.batman_model import (  # noqa: F401
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

# Attach citations to key model/fitting primitives (no wrapping; adds __references__ metadata).
get_ld_coefficients = cites(
    cite(CLARET_2018, "TESS limb darkening coefficients"),
    cite(PARVIAINEN_2015, "LDTk limb darkening toolkit"),
)(get_ld_coefficients)

compute_batman_model = cites(
    cite(MANDEL_AGOL_2002, "Analytic transit light curve model"),
    cite(KREIDBERG_2015, "batman implementation"),
)(compute_batman_model)

fit_optimize = cites(cite(BYRD_1995_LBFGSB, "L-BFGS-B bound-constrained optimization"))(
    fit_optimize
)

fit_mcmc = cites(
    cite(FOREMAN_MACKEY_2013, "emcee sampler implementation"),
    cite(GOODMAN_WEARE_2010, "affine-invariant ensemble MCMC"),
)(fit_mcmc)

quick_estimate = cites(cite(SEAGER_MALLEN_ORNELAS_2003, "Transit-shape analytic relations"))(
    quick_estimate
)

fit_transit_model = cites(
    cite(MANDEL_AGOL_2002, "Analytic transit model; Rp/Rs, a/Rs, b inference"),
    cite(KREIDBERG_2015, "batman implementation"),
    cite(FOREMAN_MACKEY_2013, "emcee for posterior sampling"),
)(fit_transit_model)

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
