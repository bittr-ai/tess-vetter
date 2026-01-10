# Module Review: `api/recovery.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is early / high leverage

Recovery is the first “active-star rescue” step agents will reach when TLS struggles.
It is very sensitive to:
- cadence/gaps (stacking + windowing)
- in-transit masking (to avoid detrending out the signal)
- unit conventions (duration_hours vs days, phase conventions)

If recovery is subtly wrong, it will either destroy real transits or fabricate detections.

## Scope (functions)

- `prepare_recovery_inputs` (multi-sector concat + transit count)
- `detrend` (builds transit mask + calls `detrend_for_recovery`)
- `stack_transits` (phase-fold + bin)
- `recover_transit_timeseries` (pure core in `recovery/pipeline.py`, but used by API)

## Audit checklist (to fill)

### Units + conventions

- [x] Period is days, t0 is BTJD days, duration is hours across API surface
- [x] Transit mask width uses a documented multiple of duration (default 1.5×) and is symmetric in phase
- [x] Stacked phase convention is clearly documented (0–1 with transit at 0.5 vs [-0.5,0.5])

### Numerical stability / edge cases

- [x] Handles gapped multi-sector time arrays without corrupting stacking bins
- [x] Excludes NaNs/Infs via `valid_mask` (LightCurve.to_internal) and propagates consistently
- [x] Clear errors when too few transits/points exist to recover

### Tests

- [x] Two-sector + gap synthetic injection recovers depth at correct order of magnitude
- [x] Recovery fails cleanly (no crash) when there are too few transits/points
- [x] Detrend mask prevents removing the injected transit (depth preserved after detrend)
