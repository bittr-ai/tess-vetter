"""WCS-aware localization via difference imaging for the public API.

Delegates to `bittr_tess_vetter.pixel.wcs_localization`.
"""

from __future__ import annotations

from bittr_tess_vetter.pixel.wcs_localization import *  # noqa: F403

# Re-export the upstream module's declared public surface.
from bittr_tess_vetter.pixel.wcs_localization import __all__  # type: ignore[attr-defined]

