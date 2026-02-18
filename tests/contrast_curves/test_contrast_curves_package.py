from __future__ import annotations

import numpy as np
import pytest

from bittr_tess_vetter.contrast_curves import (
    ContrastCurveParseError,
    combine_normalized_curves,
    parse_contrast_curve_file,
    parse_contrast_curve_with_provenance,
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


def test_parse_contrast_curve_with_provenance_prefers_fits_table(tmp_path) -> None:
    fits = pytest.importorskip("astropy.io.fits")

    path = tmp_path / "table_and_image.fits"
    image = fits.PrimaryHDU(data=np.ones((64, 64), dtype=np.float64))
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="sep_arcsec", array=[0.1, 0.5, 1.0], format="E"),
            fits.Column(name="dmag", array=[2.5, 4.5, 6.0], format="E"),
        ]
    )
    fits.HDUList([image, table]).writeto(path)

    curve, provenance = parse_contrast_curve_with_provenance(path, filter_name="Kcont")
    assert provenance["strategy"] == "fits_table"
    assert list(curve.separation_arcsec) == pytest.approx([0.1, 0.5, 1.0])
    assert list(curve.delta_mag) == pytest.approx([2.5, 4.5, 6.0])
    assert curve.filter == "Kcont"


def test_parse_contrast_curve_with_provenance_extracts_from_fits_image_header_pixscl(tmp_path) -> None:
    fits = pytest.importorskip("astropy.io.fits")
    rng = np.random.default_rng(123)
    image = rng.normal(0.0, 0.001, size=(128, 128))
    image[64, 64] += 1.0

    path = tmp_path / "tic_test_sh_562.fits"
    hdu = fits.PrimaryHDU(data=image)
    hdu.header["PIXSCL"] = 0.0183
    fits.HDUList([hdu]).writeto(path)

    curve, provenance = parse_contrast_curve_with_provenance(path)
    assert provenance["strategy"] == "fits_image_azimuthal"
    assert provenance["pixel_scale_source"] == "header:PIXSCL"
    assert provenance["pixel_scale_arcsec_per_px"] == pytest.approx(0.0183)
    assert int(provenance["annulus_used"]) >= 2
    assert len(curve.separation_arcsec) >= 2
    assert np.all(np.diff(np.asarray(curve.separation_arcsec)) > 0.0)
    assert curve.filter == "562nm"


def test_parse_contrast_curve_with_provenance_extracts_from_fits_image_lookup_scale(tmp_path) -> None:
    fits = pytest.importorskip("astropy.io.fits")
    rng = np.random.default_rng(11)
    image = rng.normal(0.0, 0.002, size=(200, 200))
    image[100, 100] += 2.0

    path = tmp_path / "TIC149302744I-at20190714_soarspeckle.fits"
    hdu = fits.PrimaryHDU(data=image)
    fits.HDUList([hdu]).writeto(path)

    curve, provenance = parse_contrast_curve_with_provenance(path)
    assert provenance["strategy"] == "fits_image_azimuthal"
    assert provenance["pixel_scale_source"] == "lookup:soar_hrcam"
    assert provenance["pixel_scale_arcsec_per_px"] == pytest.approx(0.01575)
    assert len(curve.separation_arcsec) >= 2


def test_parse_contrast_curve_with_provenance_extracts_from_fits_image_cd_matrix(tmp_path) -> None:
    fits = pytest.importorskip("astropy.io.fits")
    rng = np.random.default_rng(7)
    image = rng.normal(0.0, 0.001, size=(128, 128))
    image[64, 64] += 1.5
    path = tmp_path / "wcs_only_scale.fits"
    hdu = fits.PrimaryHDU(data=image)
    # 0.018 arcsec/pixel in degrees.
    hdu.header["CD1_1"] = -5.0e-6
    hdu.header["CD1_2"] = 0.0
    hdu.header["CD2_1"] = 0.0
    hdu.header["CD2_2"] = 5.0e-6
    fits.HDUList([hdu]).writeto(path)

    curve, provenance = parse_contrast_curve_with_provenance(path)
    assert provenance["strategy"] == "fits_image_azimuthal"
    assert provenance["pixel_scale_source"] == "header:CD_matrix"
    assert provenance["pixel_scale_arcsec_per_px"] == pytest.approx(0.018, rel=0.05)
    assert len(curve.separation_arcsec) >= 2


def test_parse_contrast_curve_with_provenance_fits_image_fails_without_pixel_scale(tmp_path) -> None:
    fits = pytest.importorskip("astropy.io.fits")
    image = np.zeros((64, 64), dtype=np.float64)
    image[32, 32] = 1.0
    path = tmp_path / "unknown_instrument.fits"
    fits.HDUList([fits.PrimaryHDU(data=image)]).writeto(path)

    with pytest.raises(ContrastCurveParseError, match="table or valid 2D image"):
        parse_contrast_curve_with_provenance(path)


def test_parse_contrast_curve_with_provenance_honors_pixel_scale_override(tmp_path) -> None:
    fits = pytest.importorskip("astropy.io.fits")
    rng = np.random.default_rng(1234)
    image = rng.normal(0.0, 0.001, size=(64, 64))
    image[32, 32] += 1.0
    path = tmp_path / "unknown_instrument.fits"
    fits.HDUList([fits.PrimaryHDU(data=image)]).writeto(path)

    curve, provenance = parse_contrast_curve_with_provenance(
        path,
        pixel_scale_arcsec_per_px=0.02,
    )
    assert provenance["pixel_scale_source"] == "override:cli"
    assert provenance["pixel_scale_arcsec_per_px"] == pytest.approx(0.02)
    assert len(curve.separation_arcsec) >= 2
