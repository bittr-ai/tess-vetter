"""Legacy module path for report build entrypoint.

This module intentionally exposes only ``build_report``.
Private build helpers now live in:
- ``tess_vetter.report._build_core``
- ``tess_vetter.report._build_panels``
- ``tess_vetter.report._build_utils``
"""

from tess_vetter.report._build_core import build_report

__all__ = ["build_report"]
