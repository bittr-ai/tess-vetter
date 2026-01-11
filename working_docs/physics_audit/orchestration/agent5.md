# Physics Audit Orchestration — Agent 5

## Assigned modules (2)

1) `src/bittr_tess_vetter/validation/checks_pixel.py`
2) `src/bittr_tess_vetter/validation/checks_catalog.py`

## Why these are top priority

These govern how evidence is produced and interpreted (pixel + catalog priors). They’re high-impact because they set failure modes and guardrails used to block/qualify candidate interpretations.

## Instructions

1) Read:
   - `working_docs/physics_audit/CONVENTIONS.md`
   - `working_docs/physics_audit/REVIEW_TEMPLATE.md`
2) Create a new module note in `working_docs/physics_audit/modules/in_progress/`:
   - Suggested filename: `validation_checks_pixel_and_catalog.md`
3) Audit checklist (minimum):
   - Units & thresholds: what each check assumes and reports (ppm/fraction, arcsec/pixels)
   - Missing-data policy: when a check is “missing” vs “failed” vs “skipped”
   - Consistency with `validation/lc_checks.py` semantics (don’t duplicate, align outputs)
   - Any crossmatch math: angular separations, proper motion handling, epoch assumptions
4) Cross-reference existing completed reviews:
   - `working_docs/physics_audit/modules/completed/api_wcs_localization.md`
   - `working_docs/physics_audit/modules/completed/api_centroid_shift.md`
   - `working_docs/physics_audit/modules/completed/api_aperture_family.md`
   - `working_docs/physics_audit/modules/completed/validation_lc_checks.md`
5) When finished:
   - Move your note to `working_docs/physics_audit/modules/completed/`.
   - Update `working_docs/physics_audit/INDEX.md` Section 4 entries for both files to ✅ and add a link.

## Deliverable

- One completed review note file covering both modules.
- `INDEX.md` updated to reflect completion.

