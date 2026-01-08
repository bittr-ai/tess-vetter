from __future__ import annotations

from unittest.mock import MagicMock, patch

from bittr_tess_vetter.api.fpp import FAST_PRESET, STANDARD_PRESET, calculate_fpp


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

