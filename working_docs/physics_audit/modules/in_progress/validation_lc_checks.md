# Module Review: `validation/lc_checks.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is early / high leverage

These are the canonical “LC-only” vetting computations (V01–V10). They run constantly and any unit mistake
(period/t0/duration) or baseline/windowing bug propagates into most downstream interpretation.

This module must remain **metrics-only**: it emits measurements + warnings; host owns any guardrail policy.

## Scope (functions)

Tier 1 (LC-only):
- `check_odd_even_depth` (V01)
- `check_secondary_eclipse` (V02)
- `check_duration_consistency` (V03)
- `check_depth_stability` (V04)
- `check_v_shape` (V05)

## Audit checklist (to fill)

### Units + conventions

- [ ] Confirm inputs are consistently: `period` days, `t0` BTJD days, `duration_hours` hours
- [ ] Confirm phase convention (transit centered at phase 0) matches `api/compute_transit.py`
- [ ] Confirm depth outputs are clearly labeled (fractional vs ppm) and consistent across checks

### Data hygiene / edge cases

- [ ] Confirm all checks honor `lightcurve.valid_mask` (no unmasked NaNs/quality)
- [ ] Confirm minimum-transit / minimum-points gating produces warnings/status rather than policy
- [ ] Confirm any “duration too long” conditions are emitted as `warnings` / `status`, not pass/fail

### Physics correctness

- [ ] V01: odd/even depth estimator matches docstring (“per-epoch median”) and sigma aggregation is correct
- [ ] V02: secondary eclipse window (0.35–0.65) is justified; baseline windows are local (not global) by default
- [ ] V03: expected duration formula and stellar-density correction are correct and units-safe
- [ ] V04: per-epoch depth fitting + chi2 calculation use consistent uncertainty inflation
- [ ] V05: trapezoid model and tF/tT estimation produce sensible values and uncertainty is computed correctly

### Statistics / uncertainty

- [ ] Identify where “confidence” is data-quantity vs signal-strength
- [ ] Confirm MAD/red-noise inflation is not being (mis)used as a policy threshold

### Tests

- [ ] Ensure existing tests cover each V01–V05 on synthetic data
- [ ] Add any missing “gap / few-point dominance” regression tests if needed

## Notes (initial)

- V03 is now metrics-only (`passed=None`) and reports `duration_ratio` + `density_corrected` + stellar params when available.
- V05 explicitly returns `status="invalid_duration"` when `duration_days >= 0.5 * period` (metrics-only).

