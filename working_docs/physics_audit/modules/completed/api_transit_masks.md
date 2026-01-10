# Module Review: `api/transit_masks.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Scope (functions)

- `count_transits`
- `compute_phase_coverage` (if present here)
- Mask constructors (phase folding / in-transit / out-of-transit)

## Audit checklist (to fill)

### Units + conventions

- [x] Phase definition explicit and consistent with `validation.base.get_in_transit_mask` (phase in [-0.5, 0.5], transit at 0)

### Numerical stability / edge cases

- [x] Works with irregular cadence and large gaps (masking is phase-based; `count_transits` uses absolute time windows)
- [x] Correct behavior for very long/short durations (validated by unit tests + min-points gates)

### Tests

- [x] Counts expected number of transits on synthetic time arrays

## Notes (initial pass)

- `api/transit_masks.py` is a thin re-export layer over `validation/base.py`; the actual physics lives in:
  - `phase_fold`, `get_in_transit_mask`, `get_out_of_transit_mask`, `measure_transit_depth`, `count_transits`, `get_odd_even_transit_indices`
- Added baseline regression tests for these primitives:
  - `tests/validation/test_transit_mask_primitives.py`
