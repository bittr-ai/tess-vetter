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


def test_localize_multi_sector_sets_insufficient_discrimination_interpretation(monkeypatch) -> None:
    from bittr_tess_vetter.api import pixel_localize as px
    from bittr_tess_vetter.pixel.tpf_fits import TPFFitsData, TPFFitsRef

    def _fake_single_sector(*, tpf_fits: TPFFitsData, **_kwargs):
        _ = tpf_fits
        return {
            "status": "ok",
            "verdict": "AMBIGUOUS",
            "raw_verdict": "ON_TARGET",
            "best_source_id": "tic:1",
            "best_source_name": "target",
            "margin": 0.4,
            "warnings": ["Low margin (0.400) between hypotheses"],
            "hypotheses_ranked": [],
            "n_in_transit": 5,
            "n_out_of_transit": 5,
            "runtime_seconds": 0.01,
            "prf_backend": "prf_lite",
            "prf_fit_diagnostics": None,
            "reliability_flagged": False,
            "reliability_flags": [],
            "diagnostics": {"n_cadences_used": 10, "n_cadences_dropped": 0},
        }

    monkeypatch.setattr(px, "localize_transit_host_single_sector_with_baseline_check", _fake_single_sector)

    n = 32
    time = np.linspace(0.0, 4.0, n, dtype=np.float64)
    flux = np.ones((n, 5, 5), dtype=np.float64)

    tpf1 = TPFFitsData(
        ref=TPFFitsRef(tic_id=1, sector=1, author="spoc"),
        time=time,
        flux=flux,
        flux_err=None,
        wcs=None,
        aperture_mask=None,
        quality=np.zeros(n, dtype=np.int32),
        camera=None,
        ccd=None,
        meta={},
    )
    tpf2 = TPFFitsData(
        ref=TPFFitsRef(tic_id=1, sector=2, author="spoc"),
        time=time,
        flux=flux,
        flux_err=None,
        wcs=None,
        aperture_mask=None,
        quality=np.zeros(n, dtype=np.int32),
        camera=None,
        ccd=None,
        meta={},
    )

    out = px.localize_transit_host_multi_sector(
        tpf_fits_list=[tpf1, tpf2],
        period_days=2.0,
        t0_btjd=0.5,
        duration_hours=2.0,
        reference_sources=[{"name": "target", "source_id": "tic:1", "row": 2.0, "col": 2.0}],
        oot_window_mult=None,
    )

    assert out["consensus"]["consensus_best_source_id"] == "tic:1"
    assert out["consensus"]["consensus_margin"] == 0.8
    assert out["consensus"]["interpretation_code"] == "INSUFFICIENT_DISCRIMINATION"
    assert out["consensus"]["reliability_flagged"] is False
