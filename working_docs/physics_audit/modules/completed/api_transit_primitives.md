# Module Review: `api/transit_primitives.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is early / high leverage

These primitives are called very early in agent workflows (cheap ephemeris-based vetting).
They also underpin higher-level LC-only checks, so unit/semantics drift here cascades.

## Scope (functions)

- `odd_even_result` (facade wrapper around `transit.vetting.compute_odd_even_result`)

## Audit checklist (to fill)

### Units + conventions

- [x] `Ephemeris.period_days` days, `t0_btjd` BTJD days, `duration_hours` hours
- [x] Uses `LightCurve.to_internal()` and respects `valid_mask`
- [x] Relative depth threshold semantics are documented (percent, not fraction)

### Statistical semantics

- [x] Odd/even depth definition matches the lc_checks epoch/parity convention (epoch boundaries between transits)
- [x] “Suspicious” is derived from a transparent, testable rule (relative depth difference)

### Edge cases

- [x] Few transits / insufficient points returns a stable, non-crashing result
- [x] NaNs/Infs are excluded via `valid_mask` (from `LightCurve.to_internal()`)
- [x] Duration too long relative to period is handled defensively

### Tests

- [x] Clean injected transit → not suspicious
- [x] Alternating depth injection → suspicious
- [x] NaNs in inputs don’t crash and don’t leak into computation
