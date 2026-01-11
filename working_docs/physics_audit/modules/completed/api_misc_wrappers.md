# Module Review: misc API wrappers

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this matters

These API modules are mostly “facade layers” that provide stable host imports and (in a few cases) attach citations to callables. The main risk is accidental API drift (missing re-export, changed symbol name) rather than physics computation (which lives in lower layers).

## Files covered

### `api/difference.py`

- Delegates to `pixel/difference.compute_difference_image` and re-exports:
  - `DifferenceImageResult`, `TransitParams`, `compute_difference_image`
- Attaches citations to `compute_difference_image` (metadata-only).
- No unit conversions are performed at this wrapper.

### `api/recovery_primitives.py`

- Re-exports recovery helpers:
  - `estimate_rotation_period`, `detrend_for_recovery`
- No additional logic.

### `api/timing_primitives.py`

- Re-exports timing primitives and attaches citations (metadata-only):
  - `measure_single_transit`, `measure_all_transit_times`, `compute_ttv_statistics`
- Unit conventions are defined in `transit/timing.py` and audited elsewhere.

### `api/sandbox_primitives.py`

- Re-exports the sandbox-safe compute surface from `compute/primitives.py`:
  - `astro`, `AstroPrimitives`, `periodogram`, `fold`, `detrend`, `box_model`
- No unit conversions or behavior changes.

### `api/primitives.py`

- Exposes the compute primitives catalog:
  - `list_primitives(include_unimplemented=False)` filters `PRIMITIVES_CATALOG` by `.implemented`.
- No unit conversions; this is metadata-only.

### `api/canonical.py` and `api/caps.py`

- Stable API re-export surfaces over:
  - `utils/canonical.py` (deterministic JSON hashing)
  - `utils/caps.py` (response size capping helpers)
- No behavior changes.

### `api/triceratops_cache.py`

- Facade re-exports around TRICERATOPS caching helpers implemented in `validation/triceratops_fpp.py`.
- No behavior changes at this layer.

## Fixes / follow-ups

No physics correctness issues identified in these wrappers.

