from __future__ import annotations

from bittr_tess_vetter.cli import contrast_curve_support as shim
from bittr_tess_vetter.contrast_curves import (
    build_ruling_summary,
    combine_normalized_curves,
    derive_contrast_verdict,
    normalize_contrast_curve,
    parse_contrast_curve_file,
)


def test_shim_reexports_reusable_primitives() -> None:
    assert shim.parse_contrast_curve_file is parse_contrast_curve_file
    assert shim.normalize_contrast_curve is normalize_contrast_curve
    assert shim.build_ruling_summary is build_ruling_summary
    assert shim.derive_contrast_verdict is derive_contrast_verdict
    assert shim.combine_normalized_curves is combine_normalized_curves

