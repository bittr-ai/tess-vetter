"""TRICERATOPS+ vendor package.

This package contains a vendored copy of TRICERATOPS+ for multi-band
False Positive Probability calculation.

Usage:
    from tess_vetter.ext.triceratops_plus_vendor.triceratops import triceratops

Note:
    TRICERATOPS+ requires optional dependencies. Install with:
        pip install tess-vetter[triceratops]
"""

# Lazy import to avoid requiring TRICERATOPS dependencies at import time
__all__ = ["triceratops"]


def __getattr__(name: str):
    """Lazy-load the triceratops module on first access."""
    if name == "triceratops":
        from tess_vetter.ext.triceratops_plus_vendor.triceratops import (
            triceratops as _tr,
        )

        return _tr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
