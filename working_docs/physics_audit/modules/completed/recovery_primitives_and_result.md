# Module Review: `recovery/primitives.py` + `recovery/result.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is high leverage

Recovery code is used specifically in “hard regimes” (active stars, strong variability) where false positives are easy to create. Small mistakes in time bases, masking, or detrending can produce convincing but artificial recovered transits.

## File: `recovery/result.py`

### Model: `StackedTransit`

- Units + conventions:
  - `phase`: `0..1` with transit centered at `0.5` (matches `stack_transits` behavior).
  - `flux`: normalized baseline ~1.0.
  - `flux_err`: fractional (same scale as `flux`).
  - `n_points_per_bin`: counts per phase bin.
  - `n_transits`: number of distinct epochs stacked.

### Model: `TrapezoidFit`

- Units + conventions:
  - `depth`: fractional depth (not ppm).
  - `duration_phase`: fraction of orbit (phase units, not days/hours).
  - `ingress_ratio`: fraction of duration (0–1).

## File: `recovery/primitives.py`

### Function: `estimate_rotation_period(time, flux, ...)`

- Units:
  - `time`: days (BTJD expected when used with TESS).
  - `min_period`/`max_period`: days.
- Method:
  - SciPy Lomb–Scargle (`signal.lombscargle`) on mean-subtracted flux.
  - Period grid is log-spaced.
  - Optional `known_period` heuristic accepts harmonics (0.5×, 2×).
- Output:
  - `(best_period_days, snr)` where `snr` is MAD-based and capped at 999.

### Function: `detrend_for_recovery(..., method=...)`

- Transit protection:
  - Requires a `transit_mask` and forwards it to detrenders so transits are not fitted away.
- Methods:
  - `"harmonic"` → `remove_stellar_variability` (requires `rotation_period`).
  - `"wotan_*"` → `compute.detrend.wotan_flatten` if installed.
- Output:
  - Returns detrended flux (normalized baseline ~1.0).

### Function: `remove_stellar_variability(...)`

- Model:
  - Fourier series in time using `sin(2π k t/P)` and `cos(2π k t/P)` terms fit on out-of-transit points.
  - Applies detrending multiplicatively: `flux / model` (preserves fractional depth better than subtraction for normalized flux).
- Numerical guardrails:
  - Returns original flux if insufficient out-of-transit points.
  - Clips `model` to `[0.5, 2.0]` to avoid division blow-ups.

### Function: `stack_transits(...)`

- Units:
  - `period`: days, `t0`: days (BTJD), `duration_hours` converted to days.
- Phase convention:
  - `phase = ((time - t0)/period) % 1`, then shifted so transit is at `0.5`.
- Default stacking window:
  - If `phase_min/max` not provided, uses a transit-centered window ~`3× duration` in phase, clamped into `[0.35, 0.65]`.
- Binning:
  - Inverse-variance weighted bin means and errors; empty bins get `flux=1.0`, `flux_err=1.0`.

### Function: `fit_trapezoid(...)`

- Space:
  - Fits in phase units (not time units).
- Guardrails:
  - Requires at least 5 valid bins (`flux_err < 0.99`).
  - Constrains duration within a factor of the initial duration by default (prevents “fit the variability” failure mode).
- Output:
  - `TrapezoidFit(depth, depth_err, duration_phase, ingress_ratio, chi2, reduced_chi2, converged)`.

### Small helpers

- `_estimate_snr`: MAD-based, capped at 999.
- `compute_detection_snr`: `depth/depth_err` capped at 999.
- `count_transits`: counts unique `floor((time - t0)/period)` epochs (assumes valid `period` and consistent time base).

## Fixes / follow-ups

No physics correctness issues identified in these modules. Main assumptions to keep documented:
- time base consistency (`t0` must match the same days scale as `time`),
- normalized flux baseline (~1.0) expected by detrending/stacking.

