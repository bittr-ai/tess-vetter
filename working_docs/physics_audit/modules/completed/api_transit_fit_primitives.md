# Module Review: `api/transit_fit_primitives.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is early / high leverage

These primitives are used any time we want a “physical model” step (even for a quick fit),
so their units, cadence handling, and starting-guess physics need to be stable and correct.

This module is mostly a stable re-export facade; the physics lives in
`transit/batman_model.py`.

## Scope (functions)

Re-exported from `transit.batman_model`:
- `detect_exposure_time`
- `quick_estimate`
- `compute_derived_parameters`
- `compute_batman_model`
- fitters: `fit_optimize`, `fit_mcmc`, `fit_transit_model`

## Audit checklist (to fill)

### Units + conventions

- [x] `detect_exposure_time` returns exposure time in **days** (batman convention)
- [x] `quick_estimate`: `depth_ppm` → `rp_rs=sqrt(depth/1e6)`, `duration_hours`, `period_days`

### Numerical stability / edge cases

- [x] `detect_exposure_time` is robust to NaNs and large gaps (multi-sector stitched data)
- [x] `quick_estimate` clips outputs to physically reasonable ranges (and documents those clips)

### API stability

- [x] Facade exports remain stable (`__all__`)
- [x] Citation metadata (`__references__`) is present on wrapped callables

### Tests

- [x] `detect_exposure_time` ignores NaNs + large gaps and recovers cadence
- [x] `quick_estimate` returns consistent `rp_rs` and sane `a_rs`/`inc` bounds
- [x] Re-exported callables expose `__references__` metadata
