from __future__ import annotations

from bittr_tess_vetter.api.sector_consistency import SectorMeasurement, compute_sector_consistency


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

