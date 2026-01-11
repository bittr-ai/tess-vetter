# Module Review: `api/detection.py` + `api/detrend.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is early / high leverage

These modules define the host-facing surface area for:
- detection results and semantics (`api/detection.py` → `domain/detection.py`), and
- detrending/normalization primitives (`api/detrend.py` → `compute/detrend.py`).

Small unit/semantics errors here (days vs hours, depth fraction vs ppm, transit masking behavior) can silently corrupt downstream vetting.

## Scope (exports)

### `src/bittr_tess_vetter/api/detection.py`

Re-exports stable domain models from `src/bittr_tess_vetter/domain/detection.py`:
- `PeriodogramPeak`
- `PeriodogramResult`
- `TransitCandidate`
- `VetterCheckResult`
- `Detection`

### `src/bittr_tess_vetter/api/detrend.py`

Re-exports detrending primitives from `src/bittr_tess_vetter/compute/detrend.py`:
- `median_detrend`
- `normalize_flux`
- `sigma_clip`
- `flatten`
- `wotan_flatten`
- `flatten_with_wotan`
- `WOTAN_AVAILABLE`

## Audit checklist (filled)

### Units + conventions

- [x] Detection models explicitly document: `period` in **days**, `t0` in **BTJD days**, `duration_hours` in **hours**
- [x] Depth conventions are explicit:
  - `TransitCandidate.depth` is **fractional** (unitless, `0 < depth <= 1`)
  - `PeriodogramPeak.depth_ppm` is **ppm** (non-negative; no upper bound enforced)
- [x] Detrend window units are explicit per function:
  - `median_detrend(window=...)` is **points** (not time)
  - `flatten(window_length=...)` is **days**
  - `wotan_flatten(window_length=...)` is **days**

### Detrending correctness + masking

- [x] `median_detrend` uses `median_filter(..., mode="reflect")`; divides `flux / baseline` and restores original NaNs
- [x] `wotan_flatten` supports a boolean `transit_mask` (True = in-transit) and passes it to wotan as `mask` (exclude from trend fit)
- [x] `flatten_with_wotan` uses wotan when available; otherwise falls back to `flatten` and logs that `transit_mask` is ignored

### Data hygiene (NaNs, gaps, ordering)

- [x] `median_detrend` interpolates through NaN gaps prior to filtering, then restores NaNs afterward
- [x] `normalize_flux` uses `np.nanmedian` and returns copies (no normalization) if median is NaN/near-zero
- [x] `flatten` and `wotan_flatten` validate time is sorted ascending; `flatten` estimates cadence via median `np.diff(time)`

### Guardrails / semantics

- [x] Pydantic validations enforce non-negative SNR/FAP and positive period/duration hours
- [~] Field names use `period` (days) rather than `period_days`; docstrings are explicit about units but aliases could be added later if host confusion appears.

## Tests

- Detrending:
  - `tests/test_compute/test_compute.py` (median detrend + normalize_flux)
  - `tests/test_compute/test_wotan_detrend.py` (wotan detrending + fallback behavior)
- Detection models:
  - No direct model-validation tests observed (models are used heavily by validation and API wrappers).

## Fixes / changes (if any)

- None required for physics correctness in these wrapper modules; follow-ups (if desired) are about API clarity (`__all__` parity, optional field aliases).
