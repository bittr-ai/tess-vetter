"""Experimental and provisional APIs.

WARNING: APIs in this module may change without notice between minor versions.
Do not rely on them for production pipelines.

These features are under active development and feedback is welcome.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "tess_vetter.api.experimental contains unstable APIs. These may change without notice.",
    UserWarning,
    stacklevel=2,
)

# Re-export experimental features
# (Add imports here as features mature)

__all__: list[str] = []
