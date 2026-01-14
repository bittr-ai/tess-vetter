"""Sandbox compute primitives API facade (host-facing).

Host applications sometimes want to expose a very small, pure-compute surface to
an execution sandbox (e.g., `python_exec` tools). The canonical implementations
live in `bittr_tess_vetter.compute.primitives`, but hosts should import from
`bittr_tess_vetter.api.*` only.
"""

from __future__ import annotations

from bittr_tess_vetter.compute.primitives import (  # noqa: F401
    AstroPrimitives,
    astro,
    box_model,
    detrend,
    fold,
    periodogram,
)
