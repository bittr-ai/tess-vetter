"""I/O-adjacent and platform-facing modules.

This package is intentionally separate from the core, array-in/array-out domain
logic. It contains filesystem/network helpers, caching, and external catalog
clients. The legacy import paths (`bittr_tess_vetter.io`, `...catalogs`,
`...network`) remain available via compatibility shims.
"""

from __future__ import annotations

__all__ = []

