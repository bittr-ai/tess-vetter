from __future__ import annotations

import numpy as np

from bittr_tess_vetter.api.negative_controls import (
    generate_control,
    generate_flux_invert,
    generate_phase_scramble,
    generate_time_scramble,
)


def test_flux_invert_is_deterministic_and_inverts_about_median() -> None:
    time = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    flux = np.array([1.0, 0.9, 1.1], dtype=np.float64)
    err = np.array([0.01, 0.01, 0.01], dtype=np.float64)

    t1, f1, e1 = generate_flux_invert(time, flux, err, seed=123)
    t2, f2, e2 = generate_flux_invert(time, flux, err, seed=999)

    assert np.allclose(t1, time)
    assert np.allclose(t2, time)
    assert np.allclose(e1, err)
    assert np.allclose(e2, err)

    med = np.nanmedian(flux)
    assert np.allclose(f1, 2.0 * med - flux)
    assert np.allclose(f2, 2.0 * med - flux)


def test_time_scramble_is_seeded_and_preserves_flux_values() -> None:
    rng = np.random.default_rng(0)
    time = np.linspace(100.0, 101.0, 50, dtype=np.float64)
    flux = rng.normal(1.0, 0.01, size=len(time)).astype(np.float64)
    err = np.full_like(time, 0.001)

    t1, f1, e1 = generate_time_scramble(time, flux, err, seed=42, block_size=7)
    t2, f2, e2 = generate_time_scramble(time, flux, err, seed=42, block_size=7)
    t3, f3, e3 = generate_time_scramble(time, flux, err, seed=43, block_size=7)

    assert np.allclose(t1, t2)
    assert np.allclose(f1, f2)
    assert np.allclose(e1, e2)

    # Different seed should generally differ.
    assert not np.allclose(f1, f3)

    # Values are preserved (permutation).
    assert np.allclose(np.sort(f1), np.sort(flux))
    assert np.allclose(np.sort(e1), np.sort(err))


def test_phase_scramble_is_seeded_and_preserves_values() -> None:
    rng = np.random.default_rng(1)
    time = np.linspace(0.0, 10.0, 200, dtype=np.float64)
    flux = rng.normal(1.0, 0.01, size=len(time)).astype(np.float64)
    err = np.full_like(time, 0.001)

    t1, f1, e1 = generate_phase_scramble(time, flux, err, period=2.0, seed=10, n_bins=8)
    t2, f2, e2 = generate_phase_scramble(time, flux, err, period=2.0, seed=10, n_bins=8)

    assert np.allclose(t1, time)
    assert np.allclose(t2, time)
    assert np.allclose(f1, f2)
    assert np.allclose(e1, e2)
    assert np.allclose(np.sort(f1), np.sort(flux))
    assert np.allclose(np.sort(e1), np.sort(err))


def test_generate_control_dispatches() -> None:
    time = np.linspace(0.0, 1.0, 20, dtype=np.float64)
    flux = np.ones_like(time)
    err = np.ones_like(time) * 0.01

    t, f, e = generate_control("time_scramble", time, flux, err, seed=0, block_size=5)
    assert len(t) == len(time)
    assert len(f) == len(time)
    assert len(e) == len(time)
