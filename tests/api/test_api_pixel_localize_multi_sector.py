from __future__ import annotations

import numpy as np


def test_localize_transit_host_multi_sector_returns_consensus_and_labels() -> None:
    from bittr_tess_vetter.api.pixel_localize import localize_transit_host_multi_sector
    from bittr_tess_vetter.pixel.tpf_fits import TPFFitsData, TPFFitsRef

    n = 200
    time = np.linspace(0.0, 10.0, n, dtype=np.float64)

    flux1 = np.ones((n, 5, 5), dtype=np.float64)
    flux1[50:60, 2, 2] -= 0.01
    tpf1 = TPFFitsData(
        ref=TPFFitsRef(tic_id=1, sector=1, author="spoc"),
        time=time,
        flux=flux1,
        flux_err=None,
        wcs=None,
        aperture_mask=None,
        quality=np.zeros(n, dtype=np.int32),
        camera=None,
        ccd=None,
        meta={},
    )

    flux2 = np.ones((n, 5, 5), dtype=np.float64)
    flux2[80:90, 2, 2] -= 0.01
    tpf2 = TPFFitsData(
        ref=TPFFitsRef(tic_id=1, sector=2, author="spoc"),
        time=time,
        flux=flux2,
        flux_err=None,
        wcs=None,
        aperture_mask=None,
        quality=np.zeros(n, dtype=np.int32),
        camera=None,
        ccd=None,
        meta={},
    )

    out = localize_transit_host_multi_sector(
        tpf_fits_list=[tpf1, tpf2],
        period_days=2.0,
        t0_btjd=0.5,
        duration_hours=2.0,
        reference_sources=[{"name": "target", "source_id": "tic:1", "row": 2.0, "col": 2.0}],
        oot_window_mult=None,
    )

    assert "per_sector_results" in out
    assert "consensus" in out
    assert len(out["per_sector_results"]) == 2
    assert out["per_sector_results"][0]["tpf_fits_ref"].startswith("tpf_fits:")
    assert out["per_sector_results"][0]["sector"] in (1, 2)
    assert "cadence_summary" in out["per_sector_results"][0]
    assert out["consensus"].get("n_sectors_total") == 2

