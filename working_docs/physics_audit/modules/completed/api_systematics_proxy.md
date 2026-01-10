# Module Review: `api/systematics.py` / `validation/systematics_proxy.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is early / high leverage

The systematics proxy check is one of the fastest “is this transit-like or pipeline-like?” diagnostics.
It feeds early triage and can gate deeper work. If it’s numerically unstable (NaNs, gaps) or uses inconsistent
masking, it can generate false “systematic” flags and waste analysis time.

## Scope (functions)

- `bittr_tess_vetter.validation.systematics_proxy.*` (feature computation)
- API/wrapper surface: `bittr_tess_vetter.api.systematics.*` (if present) and any callers in vetting pipeline

## Audit checklist (to fill)

### Inputs + conventions

- [x] Operates on normalized flux (unitless) with explicit `valid_mask` behavior
- [x] Uses the same in-transit mask convention as other checks (period days, t0 BTJD days, duration hours)
- [x] Output scores are bounded and clearly defined (higher = more systematic-like risk)

### Numerical robustness

- [x] Drops non-finite points before computing statistics/metrics (and honors `valid_mask`)
- [x] Handles gaps and irregular cadence (step metric ignores large gaps in OOT sequence)
- [x] Avoids divide-by-zero and unstable ratios (finite checks + robust sigma gating)

### Tests

- [x] Clean injected transit → low systematics proxy score (expected regime)
- [x] Step/discontinuity systematic → elevated systematics proxy score (expected regime)
- [x] NaNs/invalid points do not crash and do not dominate metrics

## Notes (final)

- `compute_systematics_proxy` now supports an optional `valid_mask` and also drops non-finite time/flux before computing masks and metrics; this prevents NaNs from silently reducing cadence counts or poisoning residual statistics.
- The step/discontinuity metric is computed on *out-of-transit* samples and uses a high-frequency noise scale (`robust_sigma(diff(oot_flux))`) so large discontinuities remain significant even when the OOT flux distribution is bimodal due to the step itself.
