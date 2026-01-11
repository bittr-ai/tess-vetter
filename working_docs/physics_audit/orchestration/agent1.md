# Physics Audit Orchestration — Agent 1

## Assigned modules (2)

1) `src/bittr_tess_vetter/api/vet.py`
2) `src/bittr_tess_vetter/api/vetting_primitives.py`

## Why these are top priority

These are the high-level vetting entry points that glue together detection outputs, light-curve checks, pixel checks, and reporting. Any unit mismatch (depth fractional vs ppm, hours vs days, BTJD vs absolute time) or mask definition bug here can silently contaminate the entire pipeline.

## Instructions

1) Read:
   - `working_docs/physics_audit/CONVENTIONS.md`
   - `working_docs/physics_audit/REVIEW_TEMPLATE.md`
2) Create a new module note in `working_docs/physics_audit/modules/in_progress/`:
   - Suggested filename: `api_vet_and_vetting_primitives.md`
3) Audit checklist (minimum):
   - Inputs/outputs: units, time base (BTJD), coordinate conventions, depth definition(s)
   - Internal conversions: hours↔days, ppm↔fractional, epoch indexing, phase folding
   - Masking: how in-transit/oot points are selected; quality/NaN handling
   - Failure modes: what happens when required evidence is missing (pixel, stellar priors, etc.)
   - Determinism/reproducibility: RNG seeds, ordering stability, cap/limits affecting results
4) Cross-reference existing completed reviews (don’t re-audit their internals, but confirm correct usage):
   - `working_docs/physics_audit/modules/completed/validation_lc_checks.md`
   - `working_docs/physics_audit/modules/completed/validation_exovetter_checks.md`
   - `working_docs/physics_audit/modules/completed/api_wcs_localization.md`
   - `working_docs/physics_audit/modules/completed/api_pixel_prf.md` (if PRF/pixel hypotheses are invoked)
5) When finished:
   - Move your note to `working_docs/physics_audit/modules/completed/`.
   - Update `working_docs/physics_audit/INDEX.md` Section 4 entries for both files to ✅ and add a link to your completed note.

## Deliverable

- One completed review note file covering both modules.
- `INDEX.md` updated to reflect completion.

