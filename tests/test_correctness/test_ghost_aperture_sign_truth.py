from __future__ import annotations

import numpy as np

from bittr_tess_vetter.validation.ghost_features import compute_ghost_features


def test_ghost_aperture_sign_consistent_false_for_opposite_sign_in_out_depths() -> None:
    # Build a tiny synthetic TPF cube where in-transit pixels behave oppositely
    # inside vs outside the aperture:
    # - aperture dims in-transit (OOT-IN > 0)
    # - annulus brightens in-transit (OOT-IN < 0)
    #
    # This should set aperture_sign_consistent=False.
    n_cadences = 4
    n_rows = 7
    n_cols = 7

    period = 10.0
    t0 = 0.0
    duration_hours = 1.0
    time = np.array([0.0, 0.01, 5.0, 5.01], dtype=np.float64)

    # Central 3x3 aperture.
    aperture_mask = np.zeros((n_rows, n_cols), dtype=bool)
    aperture_mask[2:5, 2:5] = True

    tpf = np.zeros((n_cadences, n_rows, n_cols), dtype=np.float64)

    # Identify in-transit cadences for the chosen ephemeris.
    # (We intentionally match the logic in ghost_features._compute_transit_mask.)
    phase = ((time - t0) % period) / period
    phase = np.where(phase > 0.5, phase - 1.0, phase)
    half_duration_phase = ((duration_hours / 24.0) / 2.0) / period
    in_transit = np.abs(phase) <= half_duration_phase

    # In-transit: aperture dims (negative delta), outside brightens (positive delta).
    # OOT is baseline 0.0.
    frame_in = np.zeros((n_rows, n_cols), dtype=np.float64)
    frame_in[~aperture_mask] = +1.0
    frame_in[aperture_mask] = -1.0
    tpf[in_transit] = frame_in

    feats = compute_ghost_features(
        tpf_data=tpf,
        time=time,
        aperture_mask=aperture_mask,
        period=period,
        t0=t0,
        duration_hours=duration_hours,
        tic_id=1,
        sector=1,
    )

    assert feats.in_aperture_depth > 0
    assert feats.out_aperture_depth < 0
    assert feats.aperture_sign_consistent is False
