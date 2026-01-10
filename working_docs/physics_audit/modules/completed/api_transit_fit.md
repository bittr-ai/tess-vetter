# Module Review: `api/transit_fit.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is early / high leverage

This is the host-facing “physical model” entrypoint (`fit_transit`) used by agents
once a candidate ephemeris looks plausible. If it mishandles units, masks, or
optional dependencies, it creates brittle workflows and misleading outputs.

## Scope (functions)

- `quick_estimate` (API wrapper around `transit.batman_model.quick_estimate`)
- `fit_transit` (wrapper around `transit.batman_model.fit_transit_model`)

## Audit checklist (to fill)

### Units + conventions

- [x] `Candidate.ephemeris`: `period_days` (days), `t0_btjd` (BTJD days), `duration_hours` (hours)
- [x] `t0_offset` is returned in **days** (API result field)
- [x] Stellar density conversion: solar-units → g/cm^3 via 1.41 factor

### Data hygiene

- [x] Uses `LightCurve.to_internal()` and respects `valid_mask`
- [x] Excludes NaNs/Infs before calling fit core
- [x] Errors are stable/clear when insufficient usable points exist

### Dependency behavior

- [x] If `batman` missing: returns `status="error"` with clear message
- [x] If `emcee` missing and MCMC requested: falls back to optimize

### Tests

- [x] Missing `batman` returns error result (no crash)
- [x] `valid_mask`/finite filtering is applied (NaNs don’t leak into core call)
- [x] `t0_offset` computed relative to input t0 consistently
