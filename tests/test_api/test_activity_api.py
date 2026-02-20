import numpy as np

from tess_vetter.api.activity import characterize_activity, mask_flares
from tess_vetter.api.types import LightCurve


def test_characterize_activity_recovers_rotation_period_on_sinusoid() -> None:
    cadence_seconds = 120.0
    cadence_days = cadence_seconds / 86400.0
    time = (1500.0 + np.arange(20000) * cadence_days).astype(np.float64)

    true_period = 5.0
    amp = 2e-3  # 2000 ppm
    flux = 1.0 + amp * np.sin(2.0 * np.pi * time / true_period + 0.3)
    flux_err = np.full_like(time, 2e-4)

    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    result = characterize_activity(
        lc, detect_flares=False, rotation_min_period=1.0, rotation_max_period=10.0
    )

    assert abs(result.rotation_period - true_period) / true_period < 0.05
    assert result.variability_ppm > 500


def test_characterize_activity_and_mask_flares_handles_simple_flares() -> None:
    cadence_seconds = 120.0
    cadence_days = cadence_seconds / 86400.0
    time = (1500.0 + np.arange(10000) * cadence_days).astype(np.float64)
    flux = np.ones_like(time)
    flux_err = np.full_like(time, 2e-4)

    # Inject two flare spikes (positive excursions).
    flux[2000:2005] += 0.01
    flux[7000:7010] += 0.008

    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    activity = characterize_activity(lc, detect_flares=True, flare_sigma=5.0)
    assert len(activity.flares) >= 1

    masked = mask_flares(lc, activity.flares, buffer_minutes=2.0)
    # After masking, the peak should be reduced relative to the injected spikes.
    assert float(np.nanmax(masked.flux)) < float(np.nanmax(flux))


def test_characterize_activity_is_robust_to_nans_and_unsorted_time() -> None:
    cadence_seconds = 120.0
    cadence_days = cadence_seconds / 86400.0
    time = (1500.0 + np.arange(12000) * cadence_days).astype(np.float64)
    true_period = 4.2
    flux = 1.0 + 1e-3 * np.sin(2.0 * np.pi * time / true_period)
    flux_err = np.full_like(time, 2e-4)

    # Unsort and inject NaNs.
    order = np.arange(len(time))[::-1]
    time = time[order]
    flux = flux[order]
    flux_err = flux_err[order]

    time[123] = np.nan
    flux[456] = np.nan
    flux_err[789] = np.nan

    lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
    result = characterize_activity(
        lc, detect_flares=False, rotation_min_period=1.0, rotation_max_period=10.0
    )

    assert np.isfinite(result.rotation_period)
    assert np.isfinite(result.activity_index)
