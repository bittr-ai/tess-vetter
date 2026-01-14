"""Computation primitives catalog (host-facing).

Host applications should use this module to discover which computation
primitives are available in the current installation without importing from
`bittr_tess_vetter.compute.*`.
"""

from __future__ import annotations

from bittr_tess_vetter.compute import PRIMITIVES_CATALOG, PrimitiveInfo  # noqa: F401


def list_primitives(*, include_unimplemented: bool = False) -> dict[str, PrimitiveInfo]:
    """Return the primitives catalog, optionally including unimplemented items."""
    if include_unimplemented:
        return PRIMITIVES_CATALOG.copy()
    return {k: v for k, v in PRIMITIVES_CATALOG.items() if v.implemented}


__all__ = [
    "PrimitiveInfo",
    "PRIMITIVES_CATALOG",
    "list_primitives",
]

