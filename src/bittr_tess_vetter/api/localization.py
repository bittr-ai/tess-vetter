"""Pixel-level localization diagnostics for the public API.

Delegates to `bittr_tess_vetter.pixel.localization`.

Compatibility note:
The underlying pixel-localization layer exports `LocalizationDiagnosticsResult` and
returns images as a dict. The public API historically exposed `LocalizationDiagnostics`
and `LocalizationImages`; we keep those names as aliases to avoid breaking downstream
imports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

from bittr_tess_vetter.pixel.localization import (
    LocalizationDiagnosticsResult as LocalizationDiagnostics,
)
from bittr_tess_vetter.pixel.localization import (
    TransitParams,
    compute_localization_diagnostics,
)

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

LocalizationImages: TypeAlias = dict[str, "NDArray[np.floating]"]

__all__ = [
    "LocalizationDiagnostics",
    "LocalizationImages",
    "TransitParams",
    "compute_localization_diagnostics",
]
