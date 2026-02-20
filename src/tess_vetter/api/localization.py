"""Pixel-level localization diagnostics for the public API.

Delegates to `tess_vetter.pixel.localization`.

Compatibility note:
The underlying pixel-localization layer exports `LocalizationDiagnosticsResult` and
returns images as a dict. The public API historically exposed `LocalizationDiagnostics`
and `LocalizationImages`; we keep those names as aliases to avoid breaking downstream
imports.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, TypeAlias

from tess_vetter.pixel.localization import (
    LocalizationDiagnosticsResult as LocalizationDiagnostics,
)
from tess_vetter.pixel.localization import (
    TransitParams,
    compute_localization_diagnostics,
)

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

# Avoid `dict[...]` here because Sphinx will try to render builtins `dict`
# documentation (which includes reST-invalid formatting) when documenting a
# type alias. Mapping keeps the intent while staying doc-tool friendly.
LocalizationImages: TypeAlias = Mapping[str, "NDArray[np.floating]"]

__all__ = [
    "LocalizationDiagnostics",
    "LocalizationImages",
    "TransitParams",
    "compute_localization_diagnostics",
]
