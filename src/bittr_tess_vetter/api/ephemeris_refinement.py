"""Ephemeris refinement utilities for the public API.

This module exposes deterministic local ephemeris refinement (t0 + duration)
as a stable `bittr_tess_vetter.api.*` import for host applications.

Implementation notes:
- Delegates to `bittr_tess_vetter.validation.ephemeris_refinement` to keep all
  numerical logic in one place.
"""

from __future__ import annotations

from bittr_tess_vetter.validation.ephemeris_refinement import (  # noqa: F401
    EphemerisRefinementCandidate,
    EphemerisRefinementCandidateResult,
    EphemerisRefinementConfig,
    EphemerisRefinementRunResult,
    refine_candidates_numpy,
    refine_one_candidate_numpy,
)

__all__ = [
    "EphemerisRefinementCandidate",
    "EphemerisRefinementCandidateResult",
    "EphemerisRefinementConfig",
    "EphemerisRefinementRunResult",
    "refine_one_candidate_numpy",
    "refine_candidates_numpy",
]

