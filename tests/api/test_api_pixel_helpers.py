from __future__ import annotations

import numpy as np


def test_default_cadence_mask_is_public() -> None:
    from bittr_tess_vetter.api import default_cadence_mask

    time = np.linspace(0.0, 1.0, 5, dtype=np.float64)
    flux = np.ones((5, 2, 2), dtype=np.float64)
    quality = np.zeros(5, dtype=np.int32)

    mask = default_cadence_mask(time=time, flux=flux, quality=quality, require_finite_pixels=True)
    assert mask.shape == (5,)
    assert mask.dtype == bool


def test_create_circular_aperture_mask_matches_internal() -> None:
    from bittr_tess_vetter.api.aperture import (
        _create_circular_aperture_mask,
        create_circular_aperture_mask,
    )

    shape = (5, 6)
    center_row = 2.2
    center_col = 1.7
    radius_px = 2.5

    m1 = create_circular_aperture_mask(
        shape, center_row=center_row, center_col=center_col, radius_px=radius_px
    )
    m2 = _create_circular_aperture_mask(shape=shape, radius=radius_px, center=(center_row, center_col))

    assert m1.shape == shape
    assert m1.dtype == bool
    assert np.array_equal(m1, m2)


def test_get_out_of_transit_mask_windowed_matches_pixel_impl() -> None:
    from bittr_tess_vetter.api import get_out_of_transit_mask_windowed
    from bittr_tess_vetter.pixel.aperture_family import (
        _compute_out_of_transit_mask as _compute_out_of_transit_mask_windowed,
    )

    rng = np.random.default_rng(0)
    time = np.sort(rng.uniform(0.0, 10.0, size=500).astype(np.float64))

    period = 2.0
    t0 = 0.5
    duration_hours = 2.0
    oot_margin_mult = 1.5
    oot_window_mult = 10.0

    m_api = get_out_of_transit_mask_windowed(
        time,
        period,
        t0,
        duration_hours,
        oot_margin_mult=oot_margin_mult,
        oot_window_mult=oot_window_mult,
    )
    m_px = _compute_out_of_transit_mask_windowed(
        time,
        period,
        t0,
        duration_hours / 24.0,
        oot_margin_mult=oot_margin_mult,
        oot_window_mult=oot_window_mult,
    )

    assert m_api.shape == time.shape
    assert m_api.dtype == bool
    assert np.array_equal(m_api, m_px)


def test_compute_difference_image_centroid_diagnostics_contract() -> None:
    from bittr_tess_vetter.api.wcs_localization import compute_difference_image_centroid_diagnostics
    from bittr_tess_vetter.pixel.tpf_fits import TPFFitsData, TPFFitsRef

    # Minimal synthetic TPF: constant flux with a small injected dip in one pixel "in-transit"
    n = 200
    time = np.linspace(0.0, 10.0, n, dtype=np.float64)
    flux = np.ones((n, 5, 5), dtype=np.float64)
    flux[50:60, 2, 2] -= 0.01

    tpf = TPFFitsData(
        ref=TPFFitsRef(tic_id=1, sector=1, author="spoc"),
        time=time,
        flux=flux,
        flux_err=None,
        wcs=None,  # WCS not required for diff-image centroid
        aperture_mask=None,
        quality=np.zeros(n, dtype=np.int32),
        camera=None,
        ccd=None,
        meta={},
    )

    centroid_rc, diff_image, diag = compute_difference_image_centroid_diagnostics(
        tpf_fits=tpf,
        period=2.0,
        t0=0.5,
        duration_hours=2.0,
        oot_window_mult=None,
        method="centroid",
    )

    assert diff_image.shape == (5, 5)
    assert isinstance(centroid_rc[0], float)
    assert isinstance(centroid_rc[1], float)
    assert diag["n_cadences_total"] == n
    assert diag["n_cadences_used"] == n
    assert diag["baseline_mode"] == "global"
