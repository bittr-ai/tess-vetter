# Physics Audit Orchestration — Agent 4

## Assigned modules (2)

1) `src/bittr_tess_vetter/api/io.py`
2) `src/bittr_tess_vetter/api/target.py`

## Why these are top priority

These determine “what data we’re using” and “what star we think we’re analyzing”. Name resolution, TIC/Gaia mapping, and any time/sector selection logic can silently shift the physics context (wrong star, wrong sector, wrong cadence product).

## Instructions

1) Read:
   - `working_docs/physics_audit/CONVENTIONS.md`
   - `working_docs/physics_audit/REVIEW_TEMPLATE.md`
2) Create a new module note in `working_docs/physics_audit/modules/in_progress/`:
   - Suggested filename: `api_io_and_target.md`
3) Audit checklist (minimum):
   - Time systems (BTJD/BJD) and metadata propagation
   - Sector/cadence selection rules; exposure time handling (20s vs 120s)
   - Flux type semantics (`sap` vs `pdcsap`) and normalization assumptions
   - Target resolution: ambiguity, radius thresholds, tie-breaking rules
   - Data provenance: how we record source, versions, and caching keys
4) Cross-reference existing completed reviews:
   - `working_docs/physics_audit/modules/completed/api_lightcurve.md`
   - `working_docs/physics_audit/modules/completed/api_stitch.md`
5) When finished:
   - Move your note to `working_docs/physics_audit/modules/completed/`.
   - Update `working_docs/physics_audit/INDEX.md` Section 4 entries for both files to ✅ and add a link.

## Deliverable

- One completed review note file covering both modules.
- `INDEX.md` updated to reflect completion.

