import numpy as np

from bittr_tess_vetter.api import PeriodogramResult, auto_periodogram


def test_auto_periodogram_is_exposed_via_api() -> None:
    time = np.linspace(0, 10, 500, dtype=np.float64)
    # Constant flux can yield NaNs in normalized Lomb-Scargle implementations.
    flux = 1.0 + 1e-4 * np.sin(2 * np.pi * time / 2.5)
    flux_err = np.ones_like(time) * 0.001

    result = auto_periodogram(
        time,
        flux,
        flux_err=flux_err,
        method="ls",
        min_period=1.0,
        max_period=5.0,
    )
    assert isinstance(result, PeriodogramResult)
    assert result.method == "ls"
    assert result.best_period > 0
