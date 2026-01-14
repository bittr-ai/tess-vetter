"""Tests for top-level API exports.

This module verifies:
- Golden path symbols are exported and accessible
- VettingPipeline and introspection functions are exported
- Primitives and experimental submodules exist
- All symbols in __all__ resolve correctly
"""

from __future__ import annotations

import pytest

# =============================================================================
# Golden Path Export Tests (v0.1.0 Pipeline Refactor)
# =============================================================================


def test_golden_path_core_types() -> None:
    """Golden path: Core types should be exported from api module."""
    from bittr_tess_vetter.api import (  # noqa: F401
        Candidate,
        Ephemeris,
        LightCurve,
        StellarParams,
        TPFStamp,
    )

    # Verify types are not None
    assert LightCurve is not None
    assert Ephemeris is not None
    assert Candidate is not None
    assert StellarParams is not None
    assert TPFStamp is not None


def test_golden_path_result_types() -> None:
    """Golden path: Result types should be exported (CheckResult, VettingBundleResult)."""
    from bittr_tess_vetter.api import (  # noqa: F401
        CheckResult,
        VettingBundleResult,
    )

    # Note: CheckResult from api is the dataclass version from api/types.py
    # The Pydantic version is in validation/result_schema.py (used by pipeline)
    assert CheckResult is not None
    assert VettingBundleResult is not None


def test_golden_path_entry_points() -> None:
    """Golden path: Entry points should be exported."""
    from bittr_tess_vetter.api import (  # noqa: F401
        VettingPipeline,
        calculate_fpp,
        fit_transit,
        localize_transit_source,
        run_periodogram,
        vet_candidate,
    )

    # Verify functions and classes are not None
    assert vet_candidate is not None
    assert VettingPipeline is not None
    assert run_periodogram is not None
    assert localize_transit_source is not None
    assert fit_transit is not None
    assert calculate_fpp is not None


def test_golden_path_introspection() -> None:
    """Golden path: Introspection functions should be exported."""
    from bittr_tess_vetter.api import (  # noqa: F401
        describe_checks,
        list_checks,
    )

    assert list_checks is not None
    assert describe_checks is not None

    # Verify list_checks returns a list
    from bittr_tess_vetter.validation.register_defaults import register_all_defaults
    from bittr_tess_vetter.validation.registry import CheckRegistry

    registry = CheckRegistry()
    register_all_defaults(registry)
    checks = list_checks(registry)
    assert isinstance(checks, list)
    assert len(checks) > 0


def test_golden_path_registry_types() -> None:
    """Golden path: Registry types should be exported for extensibility."""
    from bittr_tess_vetter.api import (  # noqa: F401
        CheckRegistry,
        CheckRequirements,
        CheckTier,
        PipelineConfig,
        VettingCheck,
    )

    assert CheckRegistry is not None
    assert VettingCheck is not None
    assert CheckTier is not None
    assert CheckRequirements is not None
    assert PipelineConfig is not None


def test_golden_path_aliases() -> None:
    """Golden path: Short aliases should be exported for convenience."""
    # Aliases should resolve to their targets
    from bittr_tess_vetter.api import (  # noqa: F401
        localize,  # -> localize_transit_source
        localize_transit_source,
        periodogram,  # -> run_periodogram
        run_periodogram,
        vet,  # -> vet_candidate
        vet_candidate,
    )

    assert vet is vet_candidate
    assert periodogram is run_periodogram
    assert localize is localize_transit_source


def test_golden_path_mlx_available_flag() -> None:
    """Golden path: MLX_AVAILABLE flag should always be exported."""
    from bittr_tess_vetter.api import MLX_AVAILABLE  # noqa: F401

    # Should be a boolean
    assert isinstance(MLX_AVAILABLE, bool)


# =============================================================================
# Submodule Existence Tests
# =============================================================================


def test_primitives_submodule_exists() -> None:
    """Primitives submodule should exist and export key functions."""
    from bittr_tess_vetter.api.primitives import (  # noqa: F401
        check_odd_even_depth,
        check_secondary_eclipse,
        detrend,
        fold,
    )

    assert fold is not None
    assert detrend is not None
    assert check_odd_even_depth is not None
    assert check_secondary_eclipse is not None


def test_pipeline_submodule_exists() -> None:
    """Pipeline submodule should exist and export VettingPipeline."""
    from bittr_tess_vetter.api.pipeline import (  # noqa: F401
        PipelineConfig,
        VettingPipeline,
        describe_checks,
        list_checks,
    )

    assert VettingPipeline is not None
    assert PipelineConfig is not None
    assert list_checks is not None
    assert describe_checks is not None


def test_types_submodule_exists() -> None:
    """Types submodule should exist and export API types."""
    from bittr_tess_vetter.api.types import (  # noqa: F401
        Candidate,
        CheckResult,
        Ephemeris,
        LightCurve,
        VettingBundleResult,
    )

    assert LightCurve is not None
    assert Ephemeris is not None
    assert Candidate is not None
    assert CheckResult is not None
    assert VettingBundleResult is not None


# =============================================================================
# Legacy Export Tests (retained for backward compatibility)
# =============================================================================


def test_api_top_level_exports_import() -> None:
    """This test ensures host applications can avoid deep imports."""
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

    assert hasattr(api, name), (
        f"Export {name!r} declared in __all__ but not accessible on api module"
    )
    # Also verify it's not None (indicates failed lazy import)
    obj = getattr(api, name)
    # Some symbols may legitimately be None (e.g., WOTAN_AVAILABLE=False)
    # but the attribute should exist
    assert obj is not None or name.endswith("_AVAILABLE") or name.startswith("WOTAN"), (
        f"Export {name!r} resolved to None - check import path"
    )
