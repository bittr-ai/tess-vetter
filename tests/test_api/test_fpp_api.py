from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from bittr_tess_vetter.api.fpp import (
    FAST_PRESET,
    STANDARD_PRESET,
    ContrastCurve,
    ExternalLightCurve,
    calculate_fpp,
)


def test_calculate_fpp_fast_preset_wires_expected_defaults() -> None:
    cache = MagicMock()
    with patch("bittr_tess_vetter.api.fpp.calculate_fpp_handler") as handler:
        handler.return_value = {"fpp": 0.1}
        calculate_fpp(
            cache=cache,
            tic_id=1,
            period=1.0,
            t0=0.0,
            depth_ppm=100.0,
            preset="fast",
        )
        kwargs = handler.call_args.kwargs
        assert kwargs["mc_draws"] == FAST_PRESET.mc_draws
        assert kwargs["window_duration_mult"] == FAST_PRESET.window_duration_mult
        assert kwargs["max_points"] == FAST_PRESET.max_points
        assert kwargs["min_flux_err"] == FAST_PRESET.min_flux_err
        assert kwargs["use_empirical_noise_floor"] is True


def test_calculate_fpp_standard_preset_wires_expected_defaults() -> None:
    cache = MagicMock()
    with patch("bittr_tess_vetter.api.fpp.calculate_fpp_handler") as handler:
        handler.return_value = {"fpp": 0.1}
        calculate_fpp(
            cache=cache,
            tic_id=1,
            period=1.0,
            t0=0.0,
            depth_ppm=100.0,
            preset="standard",
        )
        kwargs = handler.call_args.kwargs
        assert kwargs["mc_draws"] == STANDARD_PRESET.mc_draws
        assert kwargs["window_duration_mult"] is None
        assert kwargs["max_points"] is None
        assert kwargs["min_flux_err"] == STANDARD_PRESET.min_flux_err
        assert kwargs["use_empirical_noise_floor"] is False


def test_calculate_fpp_overrides_take_precedence() -> None:
    cache = MagicMock()
    with patch("bittr_tess_vetter.api.fpp.calculate_fpp_handler") as handler:
        handler.return_value = {"fpp": 0.1}
        calculate_fpp(
            cache=cache,
            tic_id=1,
            period=1.0,
            t0=0.0,
            depth_ppm=100.0,
            preset="fast",
            overrides={
                "mc_draws": 123,
                "window_duration_mult": None,
                "max_points": 0,
                "min_flux_err": 1e-4,
                "use_empirical_noise_floor": False,
            },
        )
        kwargs = handler.call_args.kwargs
        assert kwargs["mc_draws"] == 123
        assert kwargs["window_duration_mult"] is None
        assert kwargs["max_points"] == 0
        assert kwargs["min_flux_err"] == 1e-4
        assert kwargs["use_empirical_noise_floor"] is False


def test_external_lightcurve_dataclass_creation() -> None:
    """Test that ExternalLightCurve dataclass can be created with valid data."""
    lc = ExternalLightCurve(
        time_from_midtransit_days=np.array([-0.1, 0.0, 0.1]),
        flux=np.array([1.0, 0.99, 1.0]),
        flux_err=np.array([0.001, 0.001, 0.001]),
        filter="r",
    )
    assert lc.filter == "r"
    assert len(lc.time_from_midtransit_days) == 3
    assert len(lc.flux) == 3


def test_contrast_curve_dataclass_creation() -> None:
    """Test that ContrastCurve dataclass can be created."""
    cc = ContrastCurve(
        separation_arcsec=np.array([0.1, 0.2, 0.5, 1.0]),
        delta_mag=np.array([1.0, 3.0, 5.0, 7.0]),
        filter="Ks",
    )
    assert cc.filter == "Ks"
    assert len(cc.separation_arcsec) == 4


def test_calculate_fpp_passes_external_lightcurves_to_handler() -> None:
    """Test that external_lightcurves parameter is passed through."""
    cache = MagicMock()
    ext_lc = ExternalLightCurve(
        time_from_midtransit_days=np.array([-0.1, 0.0, 0.1]),
        flux=np.array([1.0, 0.99, 1.0]),
        flux_err=np.array([0.001, 0.001, 0.001]),
        filter="i",
    )
    with patch("bittr_tess_vetter.api.fpp.calculate_fpp_handler") as handler:
        handler.return_value = {"fpp": 0.1}
        calculate_fpp(
            cache=cache,
            tic_id=1,
            period=1.0,
            t0=0.0,
            depth_ppm=100.0,
            external_lightcurves=[ext_lc],
        )
        kwargs = handler.call_args.kwargs
        assert kwargs["external_lightcurves"] is not None
        assert len(kwargs["external_lightcurves"]) == 1
        assert kwargs["external_lightcurves"][0].filter == "i"


def test_calculate_fpp_passes_contrast_curve_to_handler() -> None:
    """Test that contrast_curve parameter is passed through."""
    cache = MagicMock()
    cc = ContrastCurve(
        separation_arcsec=np.array([0.1, 0.5, 1.0]),
        delta_mag=np.array([2.0, 5.0, 7.0]),
        filter="Ks",
    )
    with patch("bittr_tess_vetter.api.fpp.calculate_fpp_handler") as handler:
        handler.return_value = {"fpp": 0.1}
        calculate_fpp(
            cache=cache,
            tic_id=1,
            period=1.0,
            t0=0.0,
            depth_ppm=100.0,
            contrast_curve=cc,
        )
        kwargs = handler.call_args.kwargs
        assert kwargs["contrast_curve"] is not None
        assert kwargs["contrast_curve"].filter == "Ks"
