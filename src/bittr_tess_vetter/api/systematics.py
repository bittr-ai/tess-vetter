"""Systematics proxy features for the public API.

Exposes light-curve-only heuristic features meant to flag transit-like artifacts
even when pixel evidence is unavailable.

Delegates to `bittr_tess_vetter.validation.systematics_proxy`.
"""

from __future__ import annotations

from bittr_tess_vetter.api.references import PONT_2006, cite, cites
from bittr_tess_vetter.validation.systematics_proxy import (  # noqa: F401
    SystematicsProxyResult,
    compute_systematics_proxy,
)

compute_systematics_proxy = cites(
    cite(PONT_2006, "Time-correlated (red) noise as a transit false-alarm driver; motivates simple red-noise proxies"),
)(compute_systematics_proxy)

__all__ = ["SystematicsProxyResult", "compute_systematics_proxy"]
