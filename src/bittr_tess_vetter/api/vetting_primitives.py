"""Low-level vetting primitives for the public API.

These functions are useful for building diagnostics and are kept in the public
API so host apps do not import `bittr_tess_vetter.transit.*` internals.
"""

from __future__ import annotations

from bittr_tess_vetter.api.references import COUGHLIN_2016, PRSA_2011, THOMPSON_2018, cite, cites
from bittr_tess_vetter.transit.result import OddEvenResult  # noqa: F401
from bittr_tess_vetter.transit.vetting import (  # noqa: F401
    compare_odd_even_depths,
    compute_odd_even_result,
    split_odd_even,
)

# Attach citations to the exposed primitives (no wrapping; adds __references__ metadata).
split_odd_even = cites(
    cite(COUGHLIN_2016, "Odd/even parity split used for Robovetter odd/even test"),
    cite(THOMPSON_2018, "Odd/even parity split used in DR25 vetting"),
)(split_odd_even)

compare_odd_even_depths = cites(
    cite(COUGHLIN_2016, "Odd/even depth comparison statistic"),
    cite(THOMPSON_2018, "Odd/even depth comparison statistic"),
)(compare_odd_even_depths)

compute_odd_even_result = cites(
    cite(COUGHLIN_2016, "§4.2 odd/even depth test"),
    cite(THOMPSON_2018, "§3.3.1 DR25 odd/even comparison"),
    cite(PRSA_2011, "EB depth ratio context"),
)(compute_odd_even_result)

__all__ = [
    "OddEvenResult",
    "split_odd_even",
    "compare_odd_even_depths",
    "compute_odd_even_result",
]
