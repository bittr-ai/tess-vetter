"""Model competition API surface (host-facing).

This module provides a stable facade for artifact/systematics model
competition logic used by diagnostics and guardrails.
"""

from __future__ import annotations

from tess_vetter.compute.model_competition import (  # noqa: F401
    KNOWN_ARTIFACT_PERIODS,
    ArtifactPrior,
    ModelCompetitionResult,
    ModelFit,
    ModelType,
    check_period_alias,
    compute_artifact_prior,
    fit_eb_like,
    fit_transit_only,
    fit_transit_sinusoid,
    run_model_competition,
)

__all__ = [
    "ModelType",
    "ModelFit",
    "ModelCompetitionResult",
    "ArtifactPrior",
    "KNOWN_ARTIFACT_PERIODS",
    "fit_transit_only",
    "fit_transit_sinusoid",
    "fit_eb_like",
    "run_model_competition",
    "compute_artifact_prior",
    "check_period_alias",
]
