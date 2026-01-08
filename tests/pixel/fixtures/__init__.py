"""Synthetic TPF fixtures for pixel-level testing.

This module provides generators for creating synthetic Target Pixel Files (TPFs)
with known properties for testing WCS-aware localization algorithms.

Generators:
- make_synthetic_tpf_fits: Configurable star positions, transit, noise
- make_blended_binary_tpf: Two stars with configurable separation and flux ratio
- make_crowded_field_tpf: Multiple stars for crowded field scenarios
"""

from __future__ import annotations

from tests.pixel.fixtures.synthetic_cubes import (
    make_blended_binary_tpf,
    make_crowded_field_tpf,
    make_synthetic_tpf_fits,
)

__all__ = [
    "make_blended_binary_tpf",
    "make_crowded_field_tpf",
    "make_synthetic_tpf_fits",
]
