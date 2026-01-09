import numpy as np

from bittr_tess_vetter.api.periodogram import compute_transit_model as compute_transit_model_legacy
from bittr_tess_vetter.api.transit_model import compute_transit_model


def test_compute_transit_model_is_available_via_dedicated_module_and_legacy_path() -> None:
    time = np.linspace(1500.0, 1527.0, 1000, dtype=np.float64)
    flux = np.ones_like(time)
    flux_err = np.full_like(time, 1e-4)

    metrics_new = compute_transit_model(
        time=time,
        flux=flux,
        flux_err=flux_err,
        period=2.0,
        t0=1500.0,
        duration_hours=2.0,
        depth_ppm=500.0,
    )
    metrics_old = compute_transit_model_legacy(
        time=time,
        flux=flux,
        flux_err=flux_err,
        period=2.0,
        t0=1500.0,
        duration_hours=2.0,
        depth_ppm=500.0,
    )

    assert metrics_new["n_in_transit"] > 0
    assert metrics_old["n_in_transit"] == metrics_new["n_in_transit"]

