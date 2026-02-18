from __future__ import annotations

import pytest

from bittr_tess_vetter.contrast_curves import (
    ContrastCurveParseError,
    combine_normalized_curves,
    parse_contrast_curve_file,
)


def test_parse_contrast_curve_file_rejects_unsupported_extension(tmp_path) -> None:
    path = tmp_path / "curve.ps"
    path.write_text("fake", encoding="utf-8")
    with pytest.raises(ContrastCurveParseError, match="Unsupported contrast-curve extension"):
        parse_contrast_curve_file(path)


def test_combine_normalized_curves_builds_envelope() -> None:
    combined = combine_normalized_curves(
        [
            {
                "separation_arcsec": [0.5, 1.0, 2.0],
                "delta_mag": [3.0, 4.0, 5.0],
            },
            {
                "separation_arcsec": [0.5, 1.0, 2.0],
                "delta_mag": [4.0, 3.5, 4.5],
            },
        ]
    )
    assert combined is not None
    assert combined["n_observations"] == 2
    assert combined["n_points"] == 3
    assert combined["delta_mag"][0] == pytest.approx(4.0)
    assert combined["delta_mag"][2] == pytest.approx(5.0)

