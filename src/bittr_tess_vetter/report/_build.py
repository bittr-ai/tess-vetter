"""Legacy module path for report build entrypoint.

This module intentionally exposes only ``build_report``.
Private build helpers now live in:
- ``bittr_tess_vetter.report._build_core``
- ``bittr_tess_vetter.report._build_panels``
- ``bittr_tess_vetter.report._build_utils``
"""

from bittr_tess_vetter.report._build_core import build_report

__all__ = ["build_report"]
