"""Stellar activity characterization for TESS light curves.

This module provides tools for characterizing stellar activity including:
- Flare detection and cataloging
- Rotation period measurement with uncertainty
- Variability classification (spotted_rotator, pulsator, flare_star, quiet)
- Photometric activity index computation
- Flare masking for transit searches

Exports:
- Primitives: detect_flares, measure_rotation_period, classify_variability,
              compute_activity_index, mask_flares, compute_phase_amplitude,
              generate_recommendation
- Results: Flare, ActivityResult
"""

from __future__ import annotations

from bittr_tess_vetter.activity.primitives import (
    classify_variability,
    compute_activity_index,
    compute_phase_amplitude,
    detect_flares,
    generate_recommendation,
    mask_flares,
    measure_rotation_period,
)
from bittr_tess_vetter.activity.result import (
    ActivityResult,
    Flare,
)

__all__ = [
    # Primitives
    "detect_flares",
    "measure_rotation_period",
    "classify_variability",
    "compute_activity_index",
    "mask_flares",
    "compute_phase_amplitude",
    "generate_recommendation",
    # Result types
    "Flare",
    "ActivityResult",
]
