from __future__ import annotations

import importlib
from pathlib import Path
import warnings

from tess_vetter import __version__

__all__ = ["__version__"]

warnings.warn(
    "`bittr_tess_vetter` is deprecated; import `tess_vetter` instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Keep legacy submodule imports working, e.g. `bittr_tess_vetter.api`.
__path__ = [
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parent.parent / "tess_vetter"),
]


def __getattr__(name: str):
    return getattr(importlib.import_module("tess_vetter"), name)
