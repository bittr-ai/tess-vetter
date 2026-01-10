# Physics Audit Index

This is the ordered queue for the function-by-function audit.

## 0) Setup

- `CONVENTIONS.md` (units, time bases, definitions)
- `REVIEW_TEMPLATE.md` (what we check for each function)
- `INVENTORY.md` (API → implementation map; entry points used by agents first)

## 1) Early-agent entry points (review first)

### Types + contracts (prevents silent physics errors)
- `src/bittr_tess_vetter/api/types.py`
- `src/bittr_tess_vetter/api/lightcurve.py`

### Light curve assembly / normalization
- `src/bittr_tess_vetter/api/stitch.py`

### Transit masking primitives (used everywhere)
- `src/bittr_tess_vetter/api/compute_transit.py`
- `src/bittr_tess_vetter/api/transit_masks.py`
- `src/bittr_tess_vetter/validation/base.py` (mask helpers + depth measurement)

### Detection (periodogram) + minimal models
- `src/bittr_tess_vetter/api/periodogram.py`
- `src/bittr_tess_vetter/api/transit_model.py`

### LC-only vetting (fast triage)
- `src/bittr_tess_vetter/api/lc_only.py`

## 2) Next layer (review after early-agent surface is solid)

- `src/bittr_tess_vetter/api/recovery.py`
- `src/bittr_tess_vetter/api/timing.py`
- `src/bittr_tess_vetter/api/activity.py`
- `src/bittr_tess_vetter/api/fpp.py`
- `src/bittr_tess_vetter/api/pixel.py`, `src/bittr_tess_vetter/api/wcs_localization.py`

## 3) Deep pixel + PRF + advanced inference (later)

- `src/bittr_tess_vetter/api/pixel_prf.py` (and underlying `pixel/*`)
- `src/bittr_tess_vetter/transit/transit_fit.py` (batman fitting)
- `src/bittr_tess_vetter/validation/*` remaining checks

## Status legend

- ☐ Not started
- ◐ In progress
- ✅ Done
- ⚠️ Needs follow-up

