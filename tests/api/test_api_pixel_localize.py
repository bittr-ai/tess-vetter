from __future__ import annotations

import numpy as np


def test_localize_transit_host_single_sector_one_hypothesis_margin_none() -> None:
    from bittr_tess_vetter.api.pixel_localize import localize_transit_host_single_sector
    from bittr_tess_vetter.pixel.tpf_fits import TPFFitsData, TPFFitsRef

    n = 200
    time = np.linspace(0.0, 10.0, n, dtype=np.float64)
    flux = np.ones((n, 5, 5), dtype=np.float64)
    flux[50:60, 2, 2] -= 0.01

    tpf = TPFFitsData(
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

    res = localize_transit_host_single_sector(
        tpf_fits=tpf,
        period_days=2.0,
        t0_btjd=0.5,
        duration_hours=2.0,
        reference_sources=[{"name": "target", "source_id": "tic:1", "row": 2.0, "col": 2.0}],
        oot_window_mult=None,
    )

    assert res["status"] == "ok"
    assert res["margin"] is None
    assert res["verdict"] == "AMBIGUOUS"
    assert any("Only one hypothesis provided" in w for w in res.get("warnings", []))


def test_localize_single_sector_downgrades_non_physical_prf_best_fit(monkeypatch) -> None:
    from bittr_tess_vetter.api import pixel_localize as px
    from bittr_tess_vetter.pixel.tpf_fits import TPFFitsData, TPFFitsRef

    def _fake_diff_diag(**_kwargs):
        return (
            np.array([2.0, 2.0], dtype=np.float64),
            np.ones((5, 5), dtype=np.float64),
            {"n_in_transit": 5, "n_out_of_transit": 5},
        )

    def _fake_score_with_prf(_diff, _hypotheses, **_kwargs):
        return [
            {
                "source_id": "tic:1",
                "source_name": "target",
                "delta_loss": 0.0,
                "fit_amplitude": -0.2,
                "log_likelihood": 10.0,
                "fit_residual_rms": 0.1,
                "fitted_background": (0.0, 0.0, 0.0),
                "diagnostics": {"negative_flux_contribution": True},
            },
            {
                "source_id": "neighbor",
                "source_name": "neighbor",
                "delta_loss": 3.5,
                "fit_amplitude": 0.1,
            },
        ]

    monkeypatch.setattr(px, "compute_difference_image_centroid_diagnostics", _fake_diff_diag)
    monkeypatch.setattr(px, "score_hypotheses_with_prf", _fake_score_with_prf)

    n = 32
    tpf = TPFFitsData(
        ref=TPFFitsRef(tic_id=1, sector=1, author="spoc"),
        time=np.linspace(0.0, 4.0, n, dtype=np.float64),
        flux=np.ones((n, 5, 5), dtype=np.float64),
        flux_err=None,
        wcs=None,
        aperture_mask=None,
        quality=np.zeros(n, dtype=np.int32),
        camera=None,
        ccd=None,
        meta={},
    )

    res = px.localize_transit_host_single_sector(
        tpf_fits=tpf,
        period_days=2.0,
        t0_btjd=0.5,
        duration_hours=2.0,
        reference_sources=[
            {"name": "target", "source_id": "tic:1", "row": 2.0, "col": 2.0},
            {"name": "neighbor", "source_id": "neighbor", "row": 1.0, "col": 1.0},
        ],
        prf_backend="parametric",
        prf_params={"sigma_row": 1.5, "sigma_col": 1.5},
        oot_window_mult=None,
    )

    assert res["raw_verdict"] == "ON_TARGET"
    assert res["verdict"] == "AMBIGUOUS"
    assert res["reliability_flagged"] is True
    assert "NON_PHYSICAL_PRF_BEST_FIT" in res["reliability_flags"]
    assert res["interpretation_code"] == "INSUFFICIENT_DISCRIMINATION"
    assert any("Non-physical PRF best-fit indicators detected" in w for w in res["warnings"])
