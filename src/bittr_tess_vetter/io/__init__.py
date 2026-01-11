"""Compatibility shim for `bittr_tess_vetter.platform.io`."""

from __future__ import annotations

from bittr_tess_vetter.platform import io as _io

__all__ = list(_io.__all__)

for _name in __all__:
    globals()[_name] = getattr(_io, _name)

