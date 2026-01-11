# Module Review: `api/transit_fit.py` (+ `transit/batman_model.py`)

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is early / high leverage

Physical transit fitting is used to refine core planet parameters (Rp/Rs, a/Rs, inclination) and derive
secondary quantities (duration, impact parameter, stellar density). These outputs can be reused downstream
for plausibility checks and for more realistic signal modeling.

This module must remain **metrics-only** (no pass/warn/reject policy).

## Scope (files / entrypoints)

API layer:
- `src/bittr_tess_vetter/api/transit_fit.py`
  - `quick_estimate(depth_ppm, duration_hours, period_days, stellar_density_gcc=...)`
  - `fit_transit(lc, candidate, stellar, method=..., fit_limb_darkening=..., ...)`

Implementation:
- `src/bittr_tess_vetter/transit/batman_model.py`
  - `quick_estimate(...)` (analytic initialization)
  - `detect_exposure_time(time)` (cadence → batman exp_time in days)
  - `compute_derived_parameters(rp_rs, a_rs, inc, period)`
  - `fit_transit_model(...)` (requires batman; optional emcee/arviz)

## Audit checklist

### Units + conventions

- [x] `time` is BTJD days throughout (API + internal)
- [x] Candidate ephemeris uses `period_days`, `t0_btjd` in days and `duration_hours` in hours
- [x] Internal `fit_transit_model(duration=...)` expects duration in hours (not days)
- [x] `detect_exposure_time()` returns exposure time in **days** for batman

### Model + derived quantities correctness

- [x] `transit_depth_ppm` is consistent with `(rp_rs^2) * 1e6`
- [x] `impact_parameter` uses `b = (a/Rs) * cos(i)`
- [x] `duration_hours` uses a consistent T14 approximation and has safe edge-case fallback
- [x] `stellar_density_gcc` uses `rho = 3π/(G P^2) * (a/Rs)^3` with correct units

### Robustness / failure modes

- [x] Wrapper returns structured error result if `batman` is missing
- [x] Wrapper falls back to `optimize` if MCMC requested but `emcee` missing
- [x] Wrapper filters by `valid_mask`/finite and errors on insufficient usable points
- [x] Optimizer avoids non-transiting geometry minima (penalize b > 1+rp_rs)

### Tests

- [x] Wrapper tests cover missing deps + valid-mask filtering + mcmc fallback
- [x] Internal tests cover `quick_estimate`, `detect_exposure_time`, `compute_derived_parameters`
- [x] batman-dependent tests are explicitly skip-gated

## Notes (audit)

### Dependency behavior

- `api.transit_fit.fit_transit()`:
  - hard-requires `batman` at runtime (returns `status="error"` if missing)
  - soft-requires `emcee` only when `method="mcmc"` (falls back to `optimize`)

### Duration units (important)

- API candidates expose `duration_hours`; wrapper passes this through as `duration=duration_hours` into
  `transit.batman_model.fit_transit_model()`, which documents `duration` as **hours** and uses it to
  build an in/out-of-transit mask for initial depth estimation.

### Derived quantities

- `compute_derived_parameters()` follows standard relations:
  - depth: `(rp_rs^2) * 1e6`
  - impact parameter: `a_rs * cos(inc)`
  - duration: `P/pi * arcsin( sqrt((1+rp_rs)^2 - b^2) / (a_rs * sin(inc)) )` then ×24h (with fallback)
  - density: `rho = 3π/(G P^2) * (a/Rs)^3` with `P` converted to seconds and `G` in cgs

### Output normalization

- Wrapper uses `LightCurve.to_internal()` and applies `valid_mask` before fitting.
  This assumes upstream LC assembly already normalized flux to ~1; if it isn’t, the fit can still run
  but derived depth/density interpretation may not be meaningful.
