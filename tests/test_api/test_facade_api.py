from __future__ import annotations


def test_top_level_aliases_import_and_resolve() -> None:
    from bittr_tess_vetter.api import (  # noqa: F401
        aperture_family_depth_curve,
        localize,
        periodogram,
        vet,
        vet_candidate,
    )

    assert callable(vet)
    assert vet is vet_candidate
    assert callable(periodogram)
    assert callable(localize)
    assert callable(aperture_family_depth_curve)


def test_facade_imports() -> None:
    from bittr_tess_vetter.api.facade import (  # noqa: F401
        Candidate,
        Ephemeris,
        LightCurve,
        VettingBundleResult,
        analyze_ttvs,
        aperture_family_depth_curve,
        auto_periodogram,
        fit_transit,
        localize,
        measure_transit_times,
        periodogram,
        run_periodogram,
        vet,
        vet_candidate,
    )

    assert callable(vet)
    assert vet is vet_candidate
    assert callable(periodogram)
    assert callable(localize)
    assert callable(aperture_family_depth_curve)

