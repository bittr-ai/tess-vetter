"""Target and stellar parameter models (public API).

This module exposes the stable data-model contract for targets used by
host applications (e.g., MCP servers) without requiring deep imports.
"""

from __future__ import annotations

from bittr_tess_vetter.domain.target import StellarParameters, Target

__all__ = ["StellarParameters", "Target"]

