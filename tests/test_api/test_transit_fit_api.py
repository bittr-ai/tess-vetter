import builtins
import sys
import types

import numpy as np
import pytest

from tess_vetter.api.contracts import callable_input_schema_from_signature, opaque_object_schema
from tess_vetter.api.transit_fit import (
    FIT_TRANSIT_CALL_SCHEMA,
    FIT_TRANSIT_OUTPUT_SCHEMA,
    QUICK_ESTIMATE_CALL_SCHEMA,
    QUICK_ESTIMATE_OUTPUT_SCHEMA,
    TRANSIT_FIT_BOUNDARY_CONTRACT,
    TRANSIT_FIT_BOUNDARY_SCHEMA_VERSION,
    TRANSIT_FIT_MCMC_FALLBACK_METHOD,
    TRANSIT_FIT_METHODS,
    TRANSIT_FIT_MIN_USABLE_POINTS,
    TRANSIT_FIT_STATUSES,
    fit_transit,
    quick_estimate,
)
from tess_vetter.api.types import Candidate, Ephemeris, LightCurve, StellarParams


def _minimal_inputs() -> tuple[LightCurve, Candidate, StellarParams]:
    time = np.linspace(1500.0, 1510.0, 200, dtype=np.float64)
    flux = np.ones_like(time)
    flux_err = np.full_like(time, 1e-4)
    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)

    eph = Ephemeris(period_days=3.0, t0_btjd=1500.2, duration_hours=2.0)
    cand = Candidate(ephemeris=eph, depth_ppm=1000.0)
    stellar = StellarParams(teff=5800.0, logg=4.44, radius=1.0, mass=1.0, metallicity=0.0)
    return lc, cand, stellar


def test_fit_transit_missing_batman_returns_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force `import batman` to fail inside the wrapper, even if installed.
    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        if name == "batman":
            raise ImportError("batman intentionally missing for test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    lc, cand, stellar = _minimal_inputs()
    result = fit_transit(lc, cand, stellar, method="optimize")

    assert result.status == "error"
    assert "batman not installed" in (result.error_message or "")
    assert "pip install batman-package" in (result.error_message or "")


def test_fit_transit_applies_valid_mask_and_computes_t0_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Provide a fake batman module so the wrapper doesn't short-circuit.
    monkeypatch.setitem(sys.modules, "batman", types.ModuleType("batman"))

    import tess_vetter.transit.batman_model as bm

    lc, cand, stellar = _minimal_inputs()

    # Inject NaNs in inputs; LightCurve.to_internal() should mask them out.
    lc.time[5] = np.nan
    lc.flux[6] = np.nan
    lc.flux_err[7] = np.nan

    seen: dict[str, object] = {}

    def _fake_fit_transit_model(**kwargs):  # type: ignore[no-untyped-def]
        seen["time_len"] = len(kwargs["time"])
        assert np.isfinite(kwargs["time"]).all()
        assert np.isfinite(kwargs["flux"]).all()
        assert np.isfinite(kwargs["flux_err"]).all()

        t0_in = float(kwargs["t0"])

        return bm.TransitFitResult(
            fit_method="optimize",
            stellar_params=kwargs["stellar_params"],
            rp_rs=bm.ParameterEstimate(value=0.1, uncertainty=0.01),
            a_rs=bm.ParameterEstimate(value=10.0, uncertainty=1.0),
            inc=bm.ParameterEstimate(value=88.0, uncertainty=0.5),
            t0=bm.ParameterEstimate(value=t0_in + 0.123, uncertainty=0.01),
            u1=bm.ParameterEstimate(value=0.1, uncertainty=0.0),
            u2=bm.ParameterEstimate(value=0.2, uncertainty=0.0),
            transit_depth_ppm=10000.0,
            duration_hours=2.0,
            impact_parameter=0.5,
            stellar_density_gcc=1.41,
            chi_squared=1.0,
            bic=0.0,
            rms_ppm=50.0,
            phase=[],
            flux_model=[],
            flux_data=[],
            flux_err=[],
            mcmc_diagnostics=None,
            converged=True,
        )

    monkeypatch.setattr(bm, "fit_transit_model", _fake_fit_transit_model)

    result = fit_transit(lc, cand, stellar, method="optimize")
    assert int(seen["time_len"]) == len(lc.time) - 3
    assert result.status == "success"
    assert result.t0_offset == pytest.approx(0.123, rel=1e-12)


def test_fit_transit_insufficient_points_returns_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "batman", types.ModuleType("batman"))
    time = np.linspace(1500.0, 1501.0, 10, dtype=np.float64)
    flux = np.ones_like(time)
    flux_err = np.full_like(time, 1e-4)
    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    eph = Ephemeris(period_days=3.0, t0_btjd=1500.2, duration_hours=2.0)
    cand = Candidate(ephemeris=eph, depth_ppm=1000.0)
    stellar = StellarParams(teff=5800.0, logg=4.44, radius=1.0, mass=1.0, metallicity=0.0)

    result = fit_transit(lc, cand, stellar, method="optimize")
    assert result.status == "error"
    assert "Insufficient usable points" in (result.error_message or "")
    assert f"need >={TRANSIT_FIT_MIN_USABLE_POINTS}" in (result.error_message or "")


def test_fit_transit_mcmc_falls_back_to_optimize_if_emcee_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "batman", types.ModuleType("batman"))

    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        if name == "emcee":
            raise ImportError("emcee intentionally missing for test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    import tess_vetter.transit.batman_model as bm

    lc, cand, stellar = _minimal_inputs()
    seen: dict[str, object] = {}

    def _fake_fit_transit_model(**kwargs):  # type: ignore[no-untyped-def]
        seen["method"] = kwargs["method"]
        seen["duration"] = kwargs["duration"]
        assert kwargs["method"] == "optimize"
        assert kwargs["duration"] == pytest.approx(cand.ephemeris.duration_hours, rel=0)

        return bm.TransitFitResult(
            fit_method=str(kwargs["method"]),
            stellar_params=kwargs["stellar_params"],
            rp_rs=bm.ParameterEstimate(value=0.1, uncertainty=0.01),
            a_rs=bm.ParameterEstimate(value=10.0, uncertainty=1.0),
            inc=bm.ParameterEstimate(value=88.0, uncertainty=0.5),
            t0=bm.ParameterEstimate(value=float(kwargs["t0"]), uncertainty=0.01),
            u1=bm.ParameterEstimate(value=0.1, uncertainty=0.0),
            u2=bm.ParameterEstimate(value=0.2, uncertainty=0.0),
            transit_depth_ppm=10000.0,
            duration_hours=2.0,
            impact_parameter=0.5,
            stellar_density_gcc=1.41,
            chi_squared=1.0,
            bic=0.0,
            rms_ppm=50.0,
            phase=[],
            flux_model=[],
            flux_data=[],
            flux_err=[],
            mcmc_diagnostics=None,
            converged=True,
        )

    monkeypatch.setattr(bm, "fit_transit_model", _fake_fit_transit_model)

    result = fit_transit(lc, cand, stellar, method="mcmc")
    assert seen["method"] == "optimize"
    assert result.status == "success"
    assert result.fit_method == "optimize"


def test_transit_fit_boundary_contract_constants_are_stable() -> None:
    assert TRANSIT_FIT_BOUNDARY_SCHEMA_VERSION == 1
    assert TRANSIT_FIT_MIN_USABLE_POINTS == 20
    assert TRANSIT_FIT_METHODS == ("optimize", "mcmc")
    assert TRANSIT_FIT_STATUSES == ("success", "failed", "error")
    assert TRANSIT_FIT_MCMC_FALLBACK_METHOD == "optimize"


def test_transit_fit_boundary_contract_schema_is_stable() -> None:
    assert TRANSIT_FIT_BOUNDARY_CONTRACT == {
        "schema_version": 1,
        "methods": ("optimize", "mcmc"),
        "statuses": ("success", "failed", "error"),
        "default_method": "optimize",
        "mcmc_fallback_method": "optimize",
        "min_usable_points": 20,
    }


def test_transit_fit_schema_constants_track_contract_helpers() -> None:
    assert callable_input_schema_from_signature(fit_transit) == FIT_TRANSIT_CALL_SCHEMA
    assert callable_input_schema_from_signature(quick_estimate) == QUICK_ESTIMATE_CALL_SCHEMA
    assert opaque_object_schema() == FIT_TRANSIT_OUTPUT_SCHEMA
    assert opaque_object_schema() == QUICK_ESTIMATE_OUTPUT_SCHEMA


def test_transit_fit_call_schema_is_stable() -> None:
    assert FIT_TRANSIT_CALL_SCHEMA == {
        "type": "object",
        "properties": {
            "candidate": {},
            "fit_limb_darkening": {},
            "lc": {},
            "mcmc_burn": {},
            "mcmc_samples": {},
            "method": {},
            "stellar": {},
        },
        "additionalProperties": False,
        "required": ["candidate", "lc", "stellar"],
    }


def test_quick_estimate_call_schema_is_stable() -> None:
    assert QUICK_ESTIMATE_CALL_SCHEMA == {
        "type": "object",
        "properties": {
            "depth_ppm": {},
            "duration_hours": {},
            "period_days": {},
            "stellar_density_gcc": {},
        },
        "additionalProperties": False,
        "required": ["depth_ppm", "duration_hours", "period_days"],
    }
