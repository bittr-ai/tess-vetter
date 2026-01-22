from __future__ import annotations

import numpy as np

from bittr_tess_vetter.api.pixel_prf import get_prf_model
from bittr_tess_vetter.compute.pixel_timeseries import (
    TransitWindow,
    aggregate_timeseries_evidence,
    fit_all_hypotheses_timeseries,
    select_best_hypothesis_timeseries,
)


def _make_window(*, source_row: float, source_col: float) -> TransitWindow:
    n = 80
    shape = (5, 5)
    time = np.linspace(0.0, 1.0, n, dtype=np.float64)
    in_tr = (time >= 0.45) & (time <= 0.55)

    prf = get_prf_model("parametric")
    weights = prf.evaluate(source_row, source_col, shape)

    pixels = np.ones((n, shape[0], shape[1]), dtype=np.float64)
    amp = -0.01  # 1% drop
    pixels[in_tr] = pixels[in_tr] + amp * weights[None, :, :]

    errors = np.ones_like(pixels, dtype=np.float64) * 1e-3

    return TransitWindow(
        transit_idx=0,
        time=time,
        pixels=pixels,
        errors=errors,
        in_transit_mask=in_tr,
        t0_expected=0.5,
    )


def _run_hypothesis_competition(window: TransitWindow):
    prf = get_prf_model("parametric")
    hypotheses = [
        {"source_id": "target", "row": 2.0, "col": 2.0},
        {"source_id": "bg", "row": 0.0, "col": 0.0},
    ]
    fits_by_source = fit_all_hypotheses_timeseries(
        windows=[window],
        hypotheses=hypotheses,
        prf_model=prf,
        fit_baseline=True,
        baseline_order=0,
    )
    evidence = {sid: aggregate_timeseries_evidence(fits) for sid, fits in fits_by_source.items()}
    return select_best_hypothesis_timeseries(evidence, margin_threshold=2.0)


def test_pixel_timeseries_on_target_when_signal_is_target() -> None:
    w = _make_window(source_row=2.0, source_col=2.0)
    best_source_id, verdict, delta_chi2 = _run_hypothesis_competition(w)
    assert best_source_id == "target"
    assert verdict == "ON_TARGET"
    assert delta_chi2 >= 2.0


def test_pixel_timeseries_off_target_when_signal_is_background() -> None:
    w = _make_window(source_row=0.0, source_col=0.0)
    best_source_id, verdict, delta_chi2 = _run_hypothesis_competition(w)
    assert best_source_id == "bg"
    assert verdict == "OFF_TARGET"
    assert delta_chi2 >= 2.0

