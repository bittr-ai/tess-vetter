"""Low-level vetting primitives for the public API.

These functions are useful for building diagnostics and are kept in the public
API so host apps do not import `bittr_tess_vetter.transit.*` internals.
"""

from __future__ import annotations

from bittr_tess_vetter.transit.result import OddEvenResult  # noqa: F401
from bittr_tess_vetter.transit.vetting import (  # noqa: F401
    compare_odd_even_depths,
    compute_odd_even_result,
    split_odd_even,
)

__all__ = [
    "OddEvenResult",
    "split_odd_even",
    "compare_odd_even_depths",
    "compute_odd_even_result",
]

