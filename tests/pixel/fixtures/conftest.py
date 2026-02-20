"""Pytest fixtures for synthetic TPF testing.

These fixtures wrap the synthetic cube generators for convenient use in tests.
"""

from __future__ import annotations

import pytest

from tess_vetter.pixel.tpf_fits import TPFFitsData
from tests.pixel.fixtures.synthetic_cubes import (
    StarSpec,
    TransitSpec,
    make_blended_binary_tpf,
    make_crowded_field_tpf,
    make_saturated_tpf,
    make_synthetic_tpf_fits,
)


@pytest.fixture
def synth_single_star_transit() -> TPFFitsData:
    """Single star with transit at center (baseline on-target test)."""
    return make_synthetic_tpf_fits(
        shape=(500, 11, 11),
        stars=[StarSpec(row=5.0, col=5.0, flux=10000.0)],
        transit_spec=TransitSpec(
            star_idx=0,
            depth_frac=0.01,
            period=5.0,
            t0=2458001.0,
            duration_days=0.2,
        ),
        noise_level=50.0,
        seed=42,
    )


@pytest.fixture
def synth_blended_binary_3arcsec() -> TPFFitsData:
    """Close binary (3 arcsec separation) with transit on secondary."""
    return make_blended_binary_tpf(
        separation_arcsec=3.0,
        flux_ratio=0.5,
        transit_on_secondary=True,
        transit_depth_frac=0.01,
        shape=(500, 11, 11),
        noise_level=50.0,
        seed=42,
    )


@pytest.fixture
def synth_blended_binary_10arcsec() -> TPFFitsData:
    """Resolvable binary (10 arcsec separation) with transit on secondary."""
    return make_blended_binary_tpf(
        separation_arcsec=10.0,
        flux_ratio=0.5,
        transit_on_secondary=True,
        transit_depth_frac=0.01,
        shape=(500, 11, 11),
        noise_level=50.0,
        seed=42,
    )


@pytest.fixture
def synth_crowded_field_5stars() -> TPFFitsData:
    """Crowded field with 5 stars, transit on the central target."""
    return make_crowded_field_tpf(
        n_stars=5,
        transit_star_idx=0,
        transit_depth_frac=0.01,
        shape=(500, 11, 11),
        noise_level=50.0,
        seed=42,
    )


@pytest.fixture
def synth_saturated_target() -> TPFFitsData:
    """Saturated target star for saturation warning test."""
    return make_saturated_tpf(
        shape=(500, 11, 11),
        star_flux=100000.0,
        saturation_threshold=50000.0,
        transit_depth_frac=0.01,
        noise_level=50.0,
        seed=42,
    )


@pytest.fixture
def synth_low_snr_transit() -> TPFFitsData:
    """Weak signal with high noise for centroid scatter test."""
    return make_synthetic_tpf_fits(
        shape=(500, 11, 11),
        stars=[StarSpec(row=5.0, col=5.0, flux=1000.0)],  # Faint star
        transit_spec=TransitSpec(
            star_idx=0,
            depth_frac=0.005,  # Small depth
            period=5.0,
            t0=2458001.0,
            duration_days=0.2,
        ),
        noise_level=200.0,  # High noise
        seed=42,
    )


__all__ = [
    "synth_single_star_transit",
    "synth_blended_binary_3arcsec",
    "synth_blended_binary_10arcsec",
    "synth_crowded_field_5stars",
    "synth_saturated_target",
    "synth_low_snr_transit",
]
