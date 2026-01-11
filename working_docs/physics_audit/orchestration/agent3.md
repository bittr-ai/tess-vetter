# Physics Audit Orchestration — Agent 3

## Assigned modules (2)

1) `src/bittr_tess_vetter/api/tpf.py`
2) `src/bittr_tess_vetter/api/tpf_fits.py`

## Why these are top priority

TPF ingestion/representation sets the ground truth for all pixel-level diagnostics (difference images, centroid shifts, localization). FITS/WCS handling is a common source of subtle coordinate and unit errors.

## Instructions

1) Read:
   - `working_docs/physics_audit/CONVENTIONS.md`
   - `working_docs/physics_audit/REVIEW_TEMPLATE.md`
2) Create a new module note in `working_docs/physics_audit/modules/in_progress/`:
   - Suggested filename: `api_tpf_and_tpf_fits.md`
3) Audit checklist (minimum):
   - Coordinate conventions: (row,col) vs (x,y), pixel center definitions, WCS frame
   - Time base: BTJD vs BJD_TDB vs raw cadence indices; ensure consistent propagation
   - Flux units: e-/s vs counts; background subtraction conventions (if any)
   - Quality flags: what is masked/ignored; do we preserve original flags?
   - Aperture masks: indexing conventions; whether mask aligns with flux cube
4) Cross-reference existing completed reviews:
   - `working_docs/physics_audit/modules/completed/api_wcs_localization.md`
   - `working_docs/physics_audit/modules/completed/api_wcs_utils.md`
   - `working_docs/physics_audit/modules/completed/api_centroid_shift.md`
   - `working_docs/physics_audit/modules/completed/api_aperture_family.md`
5) When finished:
   - Move your note to `working_docs/physics_audit/modules/completed/`.
   - Update `working_docs/physics_audit/INDEX.md` Section 4 entries for both files to ✅ and add a link.

## Deliverable

- One completed review note file covering both modules.
- `INDEX.md` updated to reflect completion.

