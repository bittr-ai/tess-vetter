from __future__ import annotations

from tess_vetter.api.sector_consistency import SectorMeasurement, compute_sector_consistency


def test_sector_consistency_expected_scatter() -> None:
    measurements = [
        SectorMeasurement(sector=1, depth_ppm=1000.0, depth_err_ppm=50.0, quality_weight=1.0),
        SectorMeasurement(sector=2, depth_ppm=980.0, depth_err_ppm=60.0, quality_weight=1.0),
        SectorMeasurement(sector=3, depth_ppm=1020.0, depth_err_ppm=55.0, quality_weight=1.0),
    ]
    cls, outliers, pval = compute_sector_consistency(measurements)
    assert cls == "EXPECTED_SCATTER"
    assert outliers == []
    assert 0.0 <= pval <= 1.0


def test_sector_consistency_inconsistent() -> None:
    measurements = [
        SectorMeasurement(sector=1, depth_ppm=1000.0, depth_err_ppm=30.0, quality_weight=1.0),
        SectorMeasurement(sector=2, depth_ppm=980.0, depth_err_ppm=30.0, quality_weight=1.0),
        # Big deviation well beyond errors.
        SectorMeasurement(sector=3, depth_ppm=1400.0, depth_err_ppm=30.0, quality_weight=1.0),
    ]
    cls, outliers, pval = compute_sector_consistency(measurements, chi2_threshold=0.01)
    assert cls == "INCONSISTENT"
    assert 3 in outliers
    assert 0.0 <= pval <= 1.0


def test_sector_consistency_marks_nonpositive_depths_as_outliers() -> None:
    measurements = [
        SectorMeasurement(sector=1, depth_ppm=300.0, depth_err_ppm=40.0, quality_weight=1.0),
        SectorMeasurement(sector=2, depth_ppm=-5.0, depth_err_ppm=35.0, quality_weight=1.0),
        SectorMeasurement(sector=3, depth_ppm=290.0, depth_err_ppm=45.0, quality_weight=1.0),
    ]
    cls, outliers, pval = compute_sector_consistency(measurements)
    assert cls == "EXPECTED_SCATTER"
    assert 2 in outliers
    assert 0.0 <= pval <= 1.0


def test_sector_consistency_unresolvable_retains_nonpositive_outliers() -> None:
    measurements = [
        SectorMeasurement(sector=10, depth_ppm=-3.0, depth_err_ppm=50.0, quality_weight=1.0),
        SectorMeasurement(sector=11, depth_ppm=0.0, depth_err_ppm=60.0, quality_weight=1.0),
    ]
    cls, outliers, pval = compute_sector_consistency(measurements)
    assert cls == "UNRESOLVABLE"
    assert outliers == [10, 11]
    assert pval == 1.0
