from __future__ import annotations


def test_api_top_level_exports_import() -> None:
    # This test ensures host applications can avoid deep imports into
    # `bittr_tess_vetter.api.<module>` and rely on stable top-level exports.
    from bittr_tess_vetter.api import (  # noqa: F401
        ConsistencyClass,
        ControlType,
        EvidenceEnvelope,
        EvidenceProvenance,
        EphemerisEntry,
        EphemerisIndex,
        GhostFeatures,
        PhaseShiftEvent,
        SectorMeasurement,
        TPFFitsCache,
        TPFFitsRef,
        TransitTime,
        analyze_ttvs,
        classify_alias,
        compute_evidence_code_hash,
        compute_aperture_contrast,
        compute_difference_image,
        compute_edge_gradient,
        compute_ghost_features,
        compute_harmonic_scores,
        compute_prf_likeness,
        compute_reliability_curves,
        compute_sector_consistency,
        compute_secondary_significance,
        compute_spatial_uniformity,
        detect_phase_shift_events,
        generate_time_scramble,
        load_index,
    )
