"""Ephemeris specificity diagnostics for the public API.

This module exposes lightweight, deterministic diagnostics for whether an
ephemeris is "specific" (localized in phase/t0) vs. compatible with many
phase shifts (systematics/false alarm risk).

Implementation notes:
- Delegates to `bittr_tess_vetter.validation.ephemeris_specificity` to avoid
  duplicating numerical logic.
- Intended for host apps (e.g., MCP tools) that want stable imports from
  `bittr_tess_vetter.api.*` only.
"""

from __future__ import annotations

from bittr_tess_vetter.validation.ephemeris_specificity import (  # noqa: F401
    ConcentrationMetrics,
    DepthThresholdResult,
    LocalT0SensitivityResult,
    PhaseShiftNullResult,
    SmoothTemplateConfig,
    SmoothTemplateScoreResult,
    compute_concentration_metrics,
    compute_depth_threshold_numpy,
    compute_local_t0_sensitivity_numpy,
    compute_phase_shift_null,
    downsample_evenly,
    phase_shift_t0s,
    score_fixed_period_numpy,
    scores_for_t0s_numpy,
    smooth_box_template_numpy,
)

__all__ = [
    "SmoothTemplateConfig",
    "SmoothTemplateScoreResult",
    "PhaseShiftNullResult",
    "ConcentrationMetrics",
    "DepthThresholdResult",
    "LocalT0SensitivityResult",
    "downsample_evenly",
    "smooth_box_template_numpy",
    "score_fixed_period_numpy",
    "phase_shift_t0s",
    "scores_for_t0s_numpy",
    "compute_phase_shift_null",
    "compute_concentration_metrics",
    "compute_depth_threshold_numpy",
    "compute_local_t0_sensitivity_numpy",
]
