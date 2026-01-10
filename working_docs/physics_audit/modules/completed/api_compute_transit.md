# Module Review: `api/compute_transit.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is early / high leverage

Transit masking + simple depth metrics are used by almost every check (odd/even, depth stability, systematics, etc.).

## Scope (functions)

- `get_transit_mask` (duration in **hours**; consistent with `Ephemeris.duration_hours`)
- `measure_depth` (fractional depth from in/out mask)
- `fold_transit` (phase in [-0.5, 0.5])
- `detect_transit` (box model parameter measurement)

## Audit checklist (to fill)

### Units + conventions

- [x] `period` (days), `t0` (days), `duration_hours` (hours)
- [x] Depth is fractional (not ppm)

## Notes (initial pass)

`api/compute_transit.py` delegates to `compute/transit.py` and uses hours-based durations throughout for consistency with the rest of the API.

### Correctness

- [x] Mask centers transit at phase 0 consistently (phase in [-0.5, 0.5])
- [x] Consistent definition of “in-transit” window across modules (both use `abs(phase) < duration/(2*period)`)

## Cross-module consistency note

- There are two transit-mask entry points:
  - `compute/transit.get_transit_mask(time, period, t0, duration_hours)`
  - `validation/base.get_in_transit_mask(time, period, t0, duration_hours, buffer_factor=1.0)`
- They are consistent in geometry:
  - `compute/transit`: `abs(phase) < (duration_hours/24)/(2*period)`
  - `validation/base`: `abs(phase) < (duration_hours/24)/(2*period)` when `buffer_factor=1.0`

### Tests

- [x] Synthetic injection: recovered depth within tolerance

## Evidence

- Existing test coverage for `compute/transit.py`:
  - `tests/test_compute/test_compute.py::TestGetTransitMask`
  - `tests/test_compute/test_compute.py::TestMeasureDepth`
