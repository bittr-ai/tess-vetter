from __future__ import annotations

import numpy as np

from bittr_tess_vetter.pixel.aperture import ApertureDependenceResult
from bittr_tess_vetter.pixel.centroid import CentroidResult
from bittr_tess_vetter.pixel.difference import DifferenceImageResult
from bittr_tess_vetter.pixel.report import PixelVetReport, generate_pixel_vet_report


def test_generate_pixel_vet_report_bundles_metrics() -> None:
    centroid = CentroidResult(
        centroid_shift_pixels=0.3,
        significance_sigma=1.5,
        in_transit_centroid=(5.0, 5.0),
        out_of_transit_centroid=(5.2, 5.1),
        n_in_transit_cadences=20,
        n_out_transit_cadences=80,
        warnings=("LOW_TRANSIT_COUNT",),
    )
    diff = DifferenceImageResult(
        difference_image=np.zeros((11, 11)),
        localization_score=0.9,
        brightest_pixel_coords=(5, 5),
        target_coords=(5, 5),
        distance_to_target=0.0,
    )
    ap = ApertureDependenceResult(
        depths_by_aperture={1.0: 1000.0, 2.0: 1010.0},
        stability_metric=0.95,
        recommended_aperture=2.0,
        depth_variance=56.25,
        flags=("low_n_in_transit_cadences",),
        notes={},
    )

    report = generate_pixel_vet_report(centroid, diff, ap, manifest_ref="man:1")
    assert isinstance(report, PixelVetReport)
    assert report.manifest_ref == "man:1"
    assert report.centroid is not None and report.centroid["centroid_shift_pixels"] == 0.3
    assert (
        report.difference_image is not None and report.difference_image["localization_score"] == 0.9
    )
    assert (
        report.aperture_dependence is not None
        and report.aperture_dependence["stability_metric"] == 0.95
    )
    assert "LOW_TRANSIT_COUNT" in report.quality_flags
    assert "low_n_in_transit_cadences" in report.quality_flags


def test_generate_pixel_vet_report_missing_inputs_adds_flags() -> None:
    report = generate_pixel_vet_report()
    assert "MISSING_CENTROID_RESULT" in report.quality_flags
    assert "MISSING_DIFFERENCE_IMAGE_RESULT" in report.quality_flags
    assert "MISSING_APERTURE_DEPENDENCE_RESULT" in report.quality_flags
