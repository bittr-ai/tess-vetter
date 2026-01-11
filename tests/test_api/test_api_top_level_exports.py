from __future__ import annotations


def test_api_top_level_exports_import() -> None:
    # This test ensures host applications can avoid deep imports into
    # `bittr_tess_vetter.api.<module>` and rely on stable top-level exports.
    from bittr_tess_vetter.api import (  # noqa: F401
        ConsistencyClass,
        ControlType,
        EphemerisEntry,
        EphemerisIndex,
        GhostFeatures,
        PhaseShiftEvent,
        SectorMeasurement,
        compute_ghost_features,
        compute_reliability_curves,
        compute_sector_consistency,
        compute_secondary_significance,
        detect_phase_shift_events,
        generate_time_scramble,
    )

