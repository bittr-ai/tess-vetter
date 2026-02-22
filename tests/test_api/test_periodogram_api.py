import numpy as np

from tess_vetter.api import PeriodogramResult, auto_periodogram
from tess_vetter.api.periodogram import (
    AUTO_PERIODOGRAM_WRAPPER_CONTRACT,
    AUTO_PERIODOGRAM_WRAPPER_SCHEMA_VERSION,
    PERIODOGRAM_METHOD_VALUES,
    PERIODOGRAM_PRESET_VALUES,
    RUN_PERIODOGRAM_AUTO_N_PEAKS,
)


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


def test_auto_periodogram_wrapper_contract_constants_are_stable() -> None:
    assert AUTO_PERIODOGRAM_WRAPPER_SCHEMA_VERSION == 1
    assert PERIODOGRAM_PRESET_VALUES == ("fast", "thorough", "deep")
    assert PERIODOGRAM_METHOD_VALUES == ("tls", "ls", "auto")
    assert RUN_PERIODOGRAM_AUTO_N_PEAKS == 5
    assert AUTO_PERIODOGRAM_WRAPPER_CONTRACT == {
        "schema_version": AUTO_PERIODOGRAM_WRAPPER_SCHEMA_VERSION,
        "preset_values": PERIODOGRAM_PRESET_VALUES,
        "method_values": PERIODOGRAM_METHOD_VALUES,
        "forwarded_n_peaks": RUN_PERIODOGRAM_AUTO_N_PEAKS,
    }
