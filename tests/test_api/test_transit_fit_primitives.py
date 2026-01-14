import importlib.util

import numpy as np
import pytest

from bittr_tess_vetter.api import transit_fit_primitives as tfp


def test_detect_exposure_time_ignores_large_gaps_and_nans() -> None:
    cad_days = 2.0 / 60.0 / 24.0
    seg1 = 1500.0 + np.arange(1000) * cad_days
    seg2 = 1520.0 + np.arange(1000) * cad_days
    time = np.concatenate([seg1, [np.nan], seg2]).astype(np.float64)

    exposure = tfp.detect_exposure_time(time)
    assert exposure == pytest.approx(cad_days, rel=1e-6)


def test_quick_estimate_depth_maps_to_rp_rs() -> None:
    depth_ppm = 2500.0
    out = tfp.quick_estimate(depth_ppm=depth_ppm, duration_hours=2.0, period_days=3.0)
    assert out["rp_rs"] == pytest.approx(np.sqrt(depth_ppm / 1e6), rel=1e-12)
    assert 2.0 <= out["a_rs"] <= 100.0
    assert 70.0 <= out["inc"] <= 89.99


def test_transit_fit_primitives_exports_have_references_metadata() -> None:
    # Only check metadata on functions that are decorated in the facade.
    # If optional deps are missing, these functions still exist; they may fail when called.
    decorated = [
        tfp.get_ld_coefficients,
        tfp.compute_batman_model,
        tfp.fit_optimize,
        tfp.fit_mcmc,
        tfp.quick_estimate,
        tfp.fit_transit_model,
    ]
    for fn in decorated:
        assert hasattr(fn, "__references__"), f"missing __references__ on {fn.__name__}"
        refs = fn.__references__
        assert isinstance(refs, (list, tuple))
        assert len(refs) >= 1


def test_transit_fit_primitives_module_imports_without_optional_deps() -> None:
    # This is a smoke test that the facade module can be imported even if optional
    # libraries like batman/ldtk/emcee aren't installed in minimal environments.
    spec = importlib.util.find_spec("bittr_tess_vetter.api.transit_fit_primitives")
    assert spec is not None
