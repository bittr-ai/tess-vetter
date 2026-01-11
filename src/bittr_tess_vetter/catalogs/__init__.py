"""Compatibility shim for `bittr_tess_vetter.platform.catalogs`."""

from __future__ import annotations

from bittr_tess_vetter.platform import catalogs as _catalogs

__all__ = list(_catalogs.__all__)

for _name in __all__:
    globals()[_name] = getattr(_catalogs, _name)

