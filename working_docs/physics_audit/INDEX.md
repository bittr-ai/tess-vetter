# Physics Audit Index

This is the ordered queue for the function-by-function audit.

## 0) Setup

- `CONVENTIONS.md` (units, time bases, definitions)
- `REVIEW_TEMPLATE.md` (what we check for each function)
- `INVENTORY.md` (API → implementation map; entry points used by agents first)
- Module notes live in `modules/in_progress/` while active and move to `modules/completed/` when all checkboxes are checked.

## 1) Early-agent entry points (review first)

### Types + contracts (prevents silent physics errors)
- ✅ `src/bittr_tess_vetter/api/types.py` → `working_docs/physics_audit/modules/completed/api_types.md`
- ✅ `src/bittr_tess_vetter/api/lightcurve.py` → `working_docs/physics_audit/modules/completed/api_lightcurve.md`

### Light curve assembly / normalization
- ✅ `src/bittr_tess_vetter/api/stitch.py` → `working_docs/physics_audit/modules/completed/api_stitch.md`

### Transit masking primitives (used everywhere)
- ✅ `src/bittr_tess_vetter/api/compute_transit.py` → `working_docs/physics_audit/modules/completed/api_compute_transit.md`
- ✅ `src/bittr_tess_vetter/api/transit_masks.py` → `working_docs/physics_audit/modules/completed/api_transit_masks.md`
- ✅ `src/bittr_tess_vetter/validation/base.py` → `working_docs/physics_audit/modules/completed/api_validation_base_masks.md`

### Detection (periodogram) + minimal models
- ✅ `src/bittr_tess_vetter/api/periodogram.py` → `working_docs/physics_audit/modules/completed/api_periodogram.md`
- ✅ `src/bittr_tess_vetter/api/transit_model.py` → `working_docs/physics_audit/modules/completed/api_transit_model.md`

### LC-only vetting (fast triage)
- ✅ `src/bittr_tess_vetter/api/lc_only.py` → `working_docs/physics_audit/modules/completed/api_lc_only.md`

## 2) Next layer (review after early-agent surface is solid)

- ✅ `src/bittr_tess_vetter/api/recovery.py` → `working_docs/physics_audit/modules/completed/api_recovery.md`
- ✅ `src/bittr_tess_vetter/api/timing.py` → `working_docs/physics_audit/modules/completed/api_timing.md`
- ✅ `src/bittr_tess_vetter/api/activity.py` → `working_docs/physics_audit/modules/completed/api_activity.md`
- ✅ `src/bittr_tess_vetter/api/fpp.py` → `working_docs/physics_audit/modules/completed/api_fpp.md`
- ✅ `src/bittr_tess_vetter/api/pixel.py` / `src/bittr_tess_vetter/api/wcs_localization.py` → `working_docs/physics_audit/modules/completed/api_wcs_localization.md`

## 3) Deep pixel + PRF + advanced inference (later)

- ✅ `src/bittr_tess_vetter/api/pixel_prf.py` → `working_docs/physics_audit/modules/completed/api_pixel_prf.md`
- ✅ `src/bittr_tess_vetter/api/transit_fit.py` (+ `src/bittr_tess_vetter/transit/batman_model.py`) → `working_docs/physics_audit/modules/completed/api_transit_fit.md`
- ✅ `src/bittr_tess_vetter/validation/exovetter_checks.py` → `working_docs/physics_audit/modules/completed/validation_exovetter_checks.md`
- ✅ `src/bittr_tess_vetter/validation/lc_checks.py` → `working_docs/physics_audit/modules/completed/validation_lc_checks.md` (V01–V05 audited; extend later for V06–V10 as needed)

## Status legend

- ☐ Not started
- ◐ In progress
- ✅ Done
- ⚠️ Needs follow-up
