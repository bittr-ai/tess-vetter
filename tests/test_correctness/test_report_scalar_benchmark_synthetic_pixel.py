from __future__ import annotations

import numpy as np
from astropy.wcs import WCS

from bittr_tess_vetter.api.pixel import vet_pixel
from bittr_tess_vetter.api.types import Candidate, Ephemeris, TPFStamp
from bittr_tess_vetter.api.wcs_localization import (
    LocalizationVerdict,
    compute_difference_image_centroid_diagnostics,
    localize_transit_source,
)
from bittr_tess_vetter.pixel.tpf_fits import TPFFitsData, TPFFitsRef


def _to_tpf_stamp(tpf_fits: TPFFitsData) -> TPFStamp:
    return TPFStamp(
        time=np.asarray(tpf_fits.time, dtype=np.float64),
        flux=np.asarray(tpf_fits.flux, dtype=np.float64),
        flux_err=np.asarray(tpf_fits.flux_err, dtype=np.float64),
        wcs=tpf_fits.wcs,
        aperture_mask=np.asarray(tpf_fits.aperture_mask, dtype=np.int32),
        quality=np.asarray(tpf_fits.quality, dtype=np.int32),
    )


def _make_blended_binary_tpf_fits() -> TPFFitsData:
    rng = np.random.default_rng(42)
    n_cadences = 500
    n_rows = n_cols = 11
    time = np.linspace(2458000.0, 2458000.0 + 27.0, n_cadences, dtype=np.float64)
    flux = np.ones((n_cadences, n_rows, n_cols), dtype=np.float64)
    flux += rng.normal(0.0, 5e-4, size=flux.shape)

    # Primary at center, secondary offset by ~1 pixel (about 21 arcsec for TESS).
    primary_r, primary_c = 5, 5
    secondary_r, secondary_c = 6, 5
    flux[:, primary_r, primary_c] += 0.10
    flux[:, secondary_r, secondary_c] += 0.05

    period = 5.0
    t0 = 2458001.0
    duration_days = 0.2
    phase = ((time - t0) % period) / period
    phase = np.where(phase > 0.5, phase - 1.0, phase)
    in_transit = np.abs(phase) <= ((duration_days / 2.0) / period)
    flux[in_transit, secondary_r, secondary_c] *= 0.95

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [6.0, 6.0]
    wcs.wcs.crval = [120.0, -50.0]
    wcs.wcs.cdelt = [-21.0 / 3600.0, 21.0 / 3600.0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    return TPFFitsData(
        ref=TPFFitsRef(tic_id=888888888, sector=99, author="spoc"),
        time=time,
        flux=flux,
        flux_err=np.full_like(flux, 1e-3),
        wcs=wcs,
        aperture_mask=np.ones((n_rows, n_cols), dtype=np.int32),
        quality=np.zeros(n_cadences, dtype=np.int32),
        camera=1,
        ccd=1,
        meta={},
    )


def test_synthetic_pixel_scalar_benchmark_gate() -> None:
    tpf_fits = _make_blended_binary_tpf_fits()
    candidate = Candidate(
        ephemeris=Ephemeris(period_days=5.0, t0_btjd=2458001.0, duration_hours=4.8),
        depth_ppm=50000.0,
    )

    center_ra = float(tpf_fits.wcs.wcs.crval[0])
    center_dec = float(tpf_fits.wcs.wcs.crval[1])

    result = localize_transit_source(
        tpf_fits=tpf_fits,
        period=5.0,
        t0=2458001.0,
        duration_hours=4.8,
        reference_sources=[{"name": "primary", "ra": center_ra, "dec": center_dec}],
        bootstrap_draws=100,
        bootstrap_seed=42,
    )
    assert result.verdict in {
        LocalizationVerdict.ON_TARGET,
        LocalizationVerdict.OFF_TARGET,
        LocalizationVerdict.AMBIGUOUS,
    }
    assert "primary" in result.distances_to_sources
    assert float(result.distances_to_sources["primary"]) > 5.0

    _centroid_rc, _diff_image, diagnostics = compute_difference_image_centroid_diagnostics(
        tpf_fits=tpf_fits,
        period=5.0,
        t0=2458001.0,
        duration_hours=4.8,
    )
    assert diagnostics["n_cadences_used"] > 0
    assert diagnostics["n_in_transit"] > 0
    assert diagnostics["n_out_of_transit"] > 0
    assert result.extra.get("diff_peak_snr") is None or float(result.extra["diff_peak_snr"]) >= 0.0
    assert result.extra.get("peak_pixel_depth_sigma") is None or float(
        result.extra["peak_pixel_depth_sigma"]
    ) >= 0.0

    stamp = _to_tpf_stamp(tpf_fits)
    checks = vet_pixel(stamp, candidate)
    check_ids = {c.id for c in checks}
    assert {"V08", "V09", "V10"}.issubset(check_ids)
