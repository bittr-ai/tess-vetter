from __future__ import annotations

import pytest


def test_api_top_level_exports_import() -> None:
    # This test ensures host applications can avoid deep imports into
    # `bittr_tess_vetter.api.<module>` and rely on stable top-level exports.
    from bittr_tess_vetter.api import (  # noqa: F401
        ConsistencyClass,
        ControlType,
        EphemerisEntry,
        EphemerisIndex,
        EphemerisRefinementCandidate,
        EphemerisRefinementCandidateResult,
        EphemerisRefinementConfig,
        EphemerisRefinementRunResult,
        EvidenceEnvelope,
        EvidenceProvenance,
        GhostFeatures,
        PhaseShiftEvent,
        SectorMeasurement,
        TPFFitsCache,
        TPFFitsRef,
        TransitTime,
        analyze_ttvs,
        classify_alias,
        compute_aperture_contrast,
        compute_depth_correction_factor_from_flux_fraction,
        compute_difference_image,
        compute_dilution_scenarios,
        compute_edge_gradient,
        compute_evidence_code_hash,
        compute_flux_fraction_from_mag_list,
        compute_ghost_features,
        compute_harmonic_scores,
        compute_prf_likeness,
        compute_reliability_curves,
        compute_secondary_significance,
        compute_sector_consistency,
        compute_spatial_uniformity,
        compute_target_flux_fraction_from_neighbor_mags,
        detect_phase_shift_events,
        generate_time_scramble,
        load_index,
        refine_candidates_numpy,
        refine_one_candidate_numpy,
    )


# =============================================================================
# Exhaustive Export Resolution Test
# =============================================================================


def _get_api_all() -> list[str]:
    """Get the __all__ list from api module."""
    from bittr_tess_vetter import api

    return list(api.__all__)


@pytest.mark.parametrize("name", _get_api_all())
def test_all_exports_resolve(name: str) -> None:
    """Verify every symbol in __all__ is actually accessible on the api module.

    This test ensures that all 229+ symbols declared in api.__all__ can actually
    be accessed via getattr(api, name), catching any typos or missing imports.
    """
    from bittr_tess_vetter import api

    assert hasattr(api, name), f"Export {name!r} declared in __all__ but not accessible on api module"
    # Also verify it's not None (indicates failed lazy import)
    obj = getattr(api, name)
    # Some symbols may legitimately be None (e.g., WOTAN_AVAILABLE=False)
    # but the attribute should exist
    assert obj is not None or name.endswith("_AVAILABLE") or name.startswith("WOTAN"), (
        f"Export {name!r} resolved to None - check import path"
    )
