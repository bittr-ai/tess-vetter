# Physics Audit Orchestration — Agent 2

## Assigned modules (2)

1) `src/bittr_tess_vetter/api/detection.py`
2) `src/bittr_tess_vetter/api/detrend.py`

## Why these are top priority

Detection + detrending are upstream of nearly everything else. Small physics mistakes here (time base, normalization, cadence masking, period grid bounds) can create false “signals” or suppress real ones.

## Instructions

1) Read:
   - `working_docs/physics_audit/CONVENTIONS.md`
   - `working_docs/physics_audit/REVIEW_TEMPLATE.md`
2) Create a new module note in `working_docs/physics_audit/modules/in_progress/`:
   - Suggested filename: `api_detection_and_detrend.md`
3) Audit checklist (minimum):
   - Units: periods (days), durations (hours/days), flux normalization conventions
   - Detrending: window sizes and units, edge behavior, transit masking behavior
   - Detection result semantics: what is “snr”, “depth”, “duration”, “t0”, and their units/definitions
   - Guard against aliasing: harmonics, half-period, baseline/2 bounds
   - Data hygiene: quality flags, NaNs, gaps, normalization across segments
4) Cross-reference existing completed reviews:
   - `working_docs/physics_audit/modules/completed/api_periodogram.md`
   - `working_docs/physics_audit/modules/completed/api_compute_transit.md`
   - `working_docs/physics_audit/modules/completed/api_transit_masks.md`
5) When finished:
   - Move your note to `working_docs/physics_audit/modules/completed/`.
   - Update `working_docs/physics_audit/INDEX.md` Section 4 entries for both files to ✅ and add a link.

## Deliverable

- One completed review note file covering both modules.
- `INDEX.md` updated to reflect completion.

