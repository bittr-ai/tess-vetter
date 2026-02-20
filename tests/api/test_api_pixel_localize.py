from __future__ import annotations

import numpy as np


def test_localize_transit_host_single_sector_one_hypothesis_margin_none() -> None:
    from tess_vetter.api.pixel_localize import localize_transit_host_single_sector
    from tess_vetter.pixel.tpf_fits import TPFFitsData, TPFFitsRef

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
    from tess_vetter.api import pixel_localize as px
    from tess_vetter.pixel.tpf_fits import TPFFitsData, TPFFitsRef

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
        brightness_prior_enabled=False,
    )

    assert res["raw_verdict"] == "ON_TARGET"
    assert res["verdict"] == "AMBIGUOUS"
    assert res["reliability_flagged"] is True
    assert "NON_PHYSICAL_PRF_BEST_FIT" in res["reliability_flags"]
    assert res["interpretation_code"] == "INSUFFICIENT_DISCRIMINATION"
    assert any("Non-physical PRF best-fit indicators detected" in w for w in res["warnings"])
    assert res["hypotheses_ranked"][0]["fit_physical"] is False
    assert res["hypotheses_ranked"][1]["fit_physical"] is True


def test_localize_single_sector_flags_non_physical_best_fit_with_prf_lite(
    monkeypatch,
) -> None:
    from tess_vetter.api import pixel_localize as px
    from tess_vetter.pixel.tpf_fits import TPFFitsData, TPFFitsRef

    def _fake_diff_diag(**_kwargs):
        return (
            np.array([2.0, 2.0], dtype=np.float64),
            np.ones((5, 5), dtype=np.float64),
            {"n_in_transit": 4, "n_out_of_transit": 4},
        )

    def _fake_score_lite(_diff, _hypotheses, **_kwargs):
        return [
            {
                "source_id": "neighbor",
                "source_name": "neighbor",
                "delta_loss": 0.0,
                "fit_amplitude": -0.3,
            },
            {
                "source_id": "tic:1",
                "source_name": "target",
                "delta_loss": 700.0,
                "fit_amplitude": 0.2,
            },
        ]

    monkeypatch.setattr(px, "compute_difference_image_centroid_diagnostics", _fake_diff_diag)
    monkeypatch.setattr(px, "score_hypotheses_prf_lite", _fake_score_lite)

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
        prf_backend="prf_lite",
        oot_window_mult=None,
        brightness_prior_enabled=False,
    )

    assert res["raw_verdict"] == "OFF_TARGET"
    assert res["verdict"] == "AMBIGUOUS"
    assert res["reliability_flagged"] is True
    assert "NON_PHYSICAL_PRF_BEST_FIT" in res["reliability_flags"]
    assert res["interpretation_code"] == "INSUFFICIENT_DISCRIMINATION"
    assert res["hypotheses_ranked"][0]["fit_physical"] is False
    assert res["hypotheses_ranked"][1]["fit_physical"] is True


def test_baseline_sensitive_downgrade_sets_interpretation_and_reliability(
    monkeypatch,
) -> None:
    from tess_vetter.api import pixel_localize as px
    from tess_vetter.pixel.tpf_fits import TPFFitsData, TPFFitsRef

    calls = {"n": 0}

    def _fake_localize_single_sector(**_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "status": "ok",
                "verdict": "OFF_TARGET",
                "raw_verdict": "OFF_TARGET",
                "best_source_id": "neighbor",
                "best_source_name": "neighbor",
                "margin": 900.0,
                "centroid_row": 2.0,
                "centroid_col": 2.0,
                "centroid_ra_deg": 10.0,
                "centroid_dec_deg": -10.0,
                "sigma_row_pix": 1.0,
                "sigma_col_pix": 1.0,
                "sigma_arcsec": 21.0,
                "warnings": [],
                "hypotheses_ranked": [],
                "n_in_transit": 4,
                "n_out_of_transit": 4,
                "runtime_seconds": 0.1,
                "prf_backend": "prf_lite",
                "prf_fit_diagnostics": {"prf_backend": "prf_lite"},
                "reliability_flagged": False,
                "reliability_flags": [],
                "interpretation_code": None,
                "diagnostics": {},
            }
        return {
            "status": "ok",
            "verdict": "ON_TARGET",
            "raw_verdict": "ON_TARGET",
            "best_source_id": "tic:1",
            "best_source_name": "target",
            "margin": 800.0,
            "centroid_row": 3.0,
            "centroid_col": 3.0,
            "centroid_ra_deg": 10.0,
            "centroid_dec_deg": -10.0,
            "sigma_row_pix": 1.0,
            "sigma_col_pix": 1.0,
            "sigma_arcsec": 21.0,
            "warnings": [],
            "hypotheses_ranked": [],
            "n_in_transit": 4,
            "n_out_of_transit": 4,
            "runtime_seconds": 0.1,
            "prf_backend": "prf_lite",
            "prf_fit_diagnostics": {"prf_backend": "prf_lite"},
            "reliability_flagged": False,
            "reliability_flags": [],
            "interpretation_code": None,
            "diagnostics": {},
        }

    monkeypatch.setattr(px, "localize_transit_host_single_sector", _fake_localize_single_sector)

    n = 16
    tpf = TPFFitsData(
        ref=TPFFitsRef(tic_id=1, sector=1, author="spoc"),
        time=np.linspace(0.0, 2.0, n, dtype=np.float64),
        flux=np.ones((n, 5, 5), dtype=np.float64),
        flux_err=None,
        wcs=None,
        aperture_mask=None,
        quality=np.zeros(n, dtype=np.int32),
        camera=None,
        ccd=None,
        meta={},
    )

    out = px.localize_transit_host_single_sector_with_baseline_check(
        tpf_fits=tpf,
        period_days=2.0,
        t0_btjd=0.5,
        duration_hours=2.0,
        reference_sources=[{"name": "target", "source_id": "tic:1", "row": 2.0, "col": 2.0}],
        oot_window_mult=10.0,
    )

    assert out["raw_verdict"] == "OFF_TARGET"
    assert out["verdict"] == "AMBIGUOUS"
    assert out["interpretation_code"] == "INSUFFICIENT_DISCRIMINATION"
    assert out["reliability_flagged"] is True
    assert "BASELINE_SENSITIVE_LOCALIZATION" in out["reliability_flags"]


def test_brightness_prior_reorders_faint_neighbor_artifact(monkeypatch) -> None:
    from tess_vetter.api import pixel_localize as px
    from tess_vetter.pixel.tpf_fits import TPFFitsData, TPFFitsRef

    def _fake_diff_diag(**_kwargs):
        return (
            np.array([2.0, 2.0], dtype=np.float64),
            np.ones((5, 5), dtype=np.float64),
            {"n_in_transit": 6, "n_out_of_transit": 6},
        )

    def _fake_score_lite(_diff, _hypotheses, **_kwargs):
        return [
            {
                "source_id": "gaia:2",
                "source_name": "Gaia 2",
                "fit_loss": 10.0,
                "delta_loss": 0.0,
                "rank": 1,
                "fit_amplitude": 0.3,
            },
            {
                "source_id": "tic:1",
                "source_name": "Target TIC 1",
                "fit_loss": 11.0,
                "delta_loss": 1.0,
                "rank": 2,
                "fit_amplitude": 0.3,
            },
        ]

    monkeypatch.setattr(px, "compute_difference_image_centroid_diagnostics", _fake_diff_diag)
    monkeypatch.setattr(px, "score_hypotheses_prf_lite", _fake_score_lite)

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

    reference_sources = [
        {"name": "Target TIC 1", "source_id": "tic:1", "row": 2.0, "col": 2.0},
        {
            "name": "Gaia 2",
            "source_id": "gaia:2",
            "row": 1.0,
            "col": 1.0,
            "meta": {"delta_mag": 8.0},
        },
    ]

    with_prior = px.localize_transit_host_single_sector(
        tpf_fits=tpf,
        period_days=2.0,
        t0_btjd=0.5,
        duration_hours=2.0,
        reference_sources=reference_sources,
        oot_window_mult=None,
        brightness_prior_enabled=True,
        brightness_prior_weight=40.0,
        brightness_prior_softening_mag=2.5,
    )
    without_prior = px.localize_transit_host_single_sector(
        tpf_fits=tpf,
        period_days=2.0,
        t0_btjd=0.5,
        duration_hours=2.0,
        reference_sources=reference_sources,
        oot_window_mult=None,
        brightness_prior_enabled=False,
    )

    assert with_prior["best_source_id"] == "tic:1"
    assert with_prior["verdict"] == "ON_TARGET"
    assert with_prior["ranking_changed_by_prior"] is True
    assert with_prior["hypotheses_ranked"][0]["brightness_prior_penalty"] == 0.0
    assert with_prior["hypotheses_ranked"][1]["brightness_prior_penalty"] > 0.0
    assert without_prior["best_source_id"] == "gaia:2"
    assert without_prior["verdict"] == "AMBIGUOUS"
    assert without_prior["ranking_changed_by_prior"] is False
