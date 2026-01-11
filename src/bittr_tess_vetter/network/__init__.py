"""Compatibility shim for `bittr_tess_vetter.platform.network`."""

from __future__ import annotations

from bittr_tess_vetter.platform import network as _network

__all__ = list(_network.__all__)

for _name in __all__:
    globals()[_name] = getattr(_network, _name)

