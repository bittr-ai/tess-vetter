"""Negative control generators (metrics-only).

These utilities generate light-curve variants that should not contain real
transit signals. They are useful for calibrating false alarm rates and for
sanity-checking detection pipelines.

Design notes:
- Deterministic: same inputs + seed produce identical outputs.
- Open-safe: no curated assets or mission-specific policy.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np

ControlType = Literal["flux_invert", "time_scramble", "phase_scramble", "null_inject"]


def generate_flux_invert(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    *,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Invert flux around the median (dips become bumps)."""
    _ = seed  # deterministic; kept for signature stability
    median_flux = float(np.nanmedian(flux))
    flux_inverted = 2.0 * median_flux - flux
    return time.copy(), flux_inverted, flux_err.copy()


def generate_time_scramble(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    *,
    seed: int = 42,
    block_size: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scramble time order in blocks (preserve local correlations)."""
    rng = np.random.default_rng(seed)

    n_points = len(time)
    n_blocks = (n_points + block_size - 1) // block_size

    block_indices = list(range(n_blocks))
    rng.shuffle(block_indices)

    scrambled_indices: list[int] = []
    for block_idx in block_indices:
        start = block_idx * block_size
        end = min(start + block_size, n_points)
        scrambled_indices.extend(range(start, end))

    scrambled_indices_arr = np.asarray(scrambled_indices, dtype=int)

    # Replace times with a monotonic sequence to avoid discontinuities, but preserve span.
    time_scrambled = np.arange(len(time), dtype=float)
    if len(time) > 1:
        time_span = float(time[-1] - time[0])
        time_scrambled = time_scrambled / len(time_scrambled) * time_span + float(time[0])

    return (
        time_scrambled,
        np.asarray(flux)[scrambled_indices_arr],
        np.asarray(flux_err)[scrambled_indices_arr],
    )


def generate_phase_scramble(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    period: float,
    *,
    seed: int = 42,
    n_bins: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scramble flux within phase bins for a given period."""
    rng = np.random.default_rng(seed)

    phase = (time % period) / period

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(phase, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    flux_scrambled = np.asarray(flux).copy()
    flux_err_scrambled = np.asarray(flux_err).copy()

    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        if int(np.sum(mask)) > 1:
            indices = np.where(mask)[0]
            shuffled_order = rng.permutation(len(indices))
            flux_scrambled[indices] = np.asarray(flux)[indices[shuffled_order]]
            flux_err_scrambled[indices] = np.asarray(flux_err)[indices[shuffled_order]]

    return time.copy(), flux_scrambled, flux_err_scrambled


def generate_null_inject(
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    *,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return copies of the inputs unchanged."""
    _ = seed
    return time.copy(), flux.copy(), flux_err.copy()


_CONTROL_GENERATORS: dict[ControlType, Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
    "flux_invert": generate_flux_invert,
    "time_scramble": generate_time_scramble,
    "phase_scramble": generate_phase_scramble,
    "null_inject": generate_null_inject,
}


def generate_control(
    control_type: ControlType,
    time: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    *,
    seed: int = 42,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dispatch to the requested control generator."""
    if control_type not in _CONTROL_GENERATORS:
        raise ValueError(
            f"Unknown control type: {control_type}. Valid types: {list(_CONTROL_GENERATORS.keys())}"
        )
    generator = _CONTROL_GENERATORS[control_type]
    return generator(time, flux, flux_err, seed=seed, **kwargs)


__all__ = [
    "ControlType",
    "generate_control",
    "generate_flux_invert",
    "generate_null_inject",
    "generate_phase_scramble",
    "generate_time_scramble",
]

