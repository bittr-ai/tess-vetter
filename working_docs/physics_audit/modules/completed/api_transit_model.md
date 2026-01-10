# Module Review: `api/transit_model.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is early / high leverage

This is the simplest “contract sanity” helper used for:
- quick diagnostics / visualization plumbing in tool layers
- synthetic injection tests (box model baseline)
- downstream checks that need consistent in-transit masking

If its units or diagnostics are wrong, it will mislead early triage.

## Scope (functions)

- `compute_transit_model` (box model metrics)

## Audit checklist (to fill)

### Units + conventions

- [x] `time` and `t0` are in BTJD days
- [x] `period` is in days
- [x] `duration_hours` is in hours (converted internally to days for phasing)
- [x] `depth_ppm` converted to fractional depth (`depth_ppm / 1e6`) consistently

### Numerical stability / edge cases

- [x] NaNs/Infs are excluded from diagnostics (`finite_mask`)
- [x] `n_in_transit` uses the same in-transit definition as the underlying box model
- [x] Reduced chi-square uses a correct degrees-of-freedom assumption for “no-fit” diagnostics (chi2 per finite point)

### Tests

- [x] Perfect synthetic box model → `rms_residual≈0`, `chi2≈0`
- [x] NaNs are ignored but still produce stable outputs
- [x] Insufficient finite points raises a clear error
