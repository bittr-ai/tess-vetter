"""Low-level transit timing primitives for the public API.

This module exists for host applications that need stable access to timing
building blocks (single-transit fits, batch fitting) without importing
`tess_vetter.transit.*` internals directly.
"""

from __future__ import annotations

from tess_vetter.api.references import (
    AGOL_2005,
    BYRD_1995_LBFGSB,
    HOLMAN_MURRAY_2005,
    IVSHINA_WINN_2022,
    cite,
    cites,
)
from tess_vetter.transit.timing import (  # noqa: F401
    compute_ttv_statistics,
    measure_all_transit_times,
    measure_single_transit,
)

measure_single_transit = cites(
    cite(BYRD_1995_LBFGSB, "L-BFGS-B bound-constrained trapezoid fit"),
    cite(IVSHINA_WINN_2022, "Template fitting/transit timing methodology context"),
)(measure_single_transit)

measure_all_transit_times = cites(
    cite(IVSHINA_WINN_2022, "Batch transit time measurement from TESS light curves"),
)(measure_all_transit_times)

compute_ttv_statistics = cites(
    cite(HOLMAN_MURRAY_2005, "Foundational TTV concept"),
    cite(AGOL_2005, "TTV sensitivity and methods"),
)(compute_ttv_statistics)

__all__ = [
    "measure_single_transit",
    "measure_all_transit_times",
    "compute_ttv_statistics",
]
