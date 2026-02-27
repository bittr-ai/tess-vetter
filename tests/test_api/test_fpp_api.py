from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from tess_vetter.api.contracts import callable_input_schema_from_signature, opaque_object_schema
from tess_vetter.api.fpp import (
    CALCULATE_FPP_CALL_SCHEMA,
    CALCULATE_FPP_OUTPUT_SCHEMA,
    DEFAULT_BIN_ERR,
    DEFAULT_BIN_STAT,
    DEFAULT_MC_DRAWS,
    DEFAULT_MIN_FLUX_ERR,
    DEFAULT_POINT_REDUCTION,
    DEFAULT_TARGET_POINTS,
    DEFAULT_USE_EMPIRICAL_NOISE_FLOOR,
    DEFAULT_WINDOW_DURATION_MULT,
    ContrastCurve,
    ExternalLightCurve,
    calculate_fpp,
)


def test_calculate_fpp_knob_defaults_wire_expected_values() -> None:
    cache = MagicMock()
    with patch("tess_vetter.api.fpp.calculate_fpp_handler") as handler:
        handler.return_value = {"fpp": 0.1}
        calculate_fpp(
            cache=cache,
            tic_id=1,
            period=1.0,
            t0=0.0,
            depth_ppm=100.0,
        )
        kwargs = handler.call_args.kwargs
        assert kwargs["mc_draws"] == DEFAULT_MC_DRAWS
        assert kwargs["window_duration_mult"] == DEFAULT_WINDOW_DURATION_MULT
        assert kwargs["point_reduction"] == DEFAULT_POINT_REDUCTION
        assert kwargs["target_points"] == DEFAULT_TARGET_POINTS
        assert kwargs["bin_stat"] == DEFAULT_BIN_STAT
        assert kwargs["bin_err"] == DEFAULT_BIN_ERR
        assert kwargs["max_points"] is None
        assert kwargs["min_flux_err"] == DEFAULT_MIN_FLUX_ERR
        assert kwargs["use_empirical_noise_floor"] is DEFAULT_USE_EMPIRICAL_NOISE_FLOOR


def test_calculate_fpp_explicit_knobs_override_defaults() -> None:
    cache = MagicMock()
    with patch("tess_vetter.api.fpp.calculate_fpp_handler") as handler:
        handler.return_value = {"fpp": 0.1}
        calculate_fpp(
            cache=cache,
            tic_id=1,
            period=1.0,
            t0=0.0,
            depth_ppm=100.0,
            mc_draws=123,
            window_duration_mult=None,
            point_reduction="bin",
            target_points=250,
            bin_stat="mean",
            bin_err="propagate",
            max_points=250,
            min_flux_err=1e-4,
            use_empirical_noise_floor=False,
            drop_scenario=["SEB"],
        )
        kwargs = handler.call_args.kwargs
        assert kwargs["mc_draws"] == 123
        assert kwargs["window_duration_mult"] is None
        assert kwargs["point_reduction"] == "bin"
        assert kwargs["target_points"] == 250
        assert kwargs["max_points"] == 250
        assert kwargs["min_flux_err"] == 1e-4
        assert kwargs["use_empirical_noise_floor"] is False
        assert kwargs["drop_scenario"] == ["SEB"]


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
    with patch("tess_vetter.api.fpp.calculate_fpp_handler") as handler:
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
    with patch("tess_vetter.api.fpp.calculate_fpp_handler") as handler:
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


def test_calculate_fpp_schema_constants_track_contract_helpers() -> None:
    assert callable_input_schema_from_signature(calculate_fpp) == CALCULATE_FPP_CALL_SCHEMA
    assert opaque_object_schema() == CALCULATE_FPP_OUTPUT_SCHEMA


def test_calculate_fpp_call_schema_is_stable() -> None:
    assert CALCULATE_FPP_CALL_SCHEMA == {
        "type": "object",
        "properties": {
            "allow_network": {},
            "bin_err": {},
            "bin_stat": {},
            "cache": {},
            "contrast_curve": {},
            "depth_ppm": {},
            "drop_scenario": {},
            "duration_hours": {},
            "external_lightcurves": {},
            "max_points": {},
            "mc_draws": {},
            "min_flux_err": {},
            "overrides": {},
            "period": {},
            "point_reduction": {},
            "progress_hook": {},
            "replicates": {},
            "seed": {},
            "sectors": {},
            "stellar_mass": {},
            "stellar_radius": {},
            "t0": {},
            "target_points": {},
            "tic_id": {},
            "timeout_seconds": {},
            "tmag": {},
            "use_empirical_noise_floor": {},
            "window_duration_mult": {},
        },
        "additionalProperties": False,
        "required": ["cache", "depth_ppm", "period", "t0", "tic_id"],
    }
