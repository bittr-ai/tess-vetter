# Module Review: `transit/timing.py` + `transit/result.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is high leverage

Timing and odd/even vetting are “convincing failure modes”: they can generate plausible-looking diagnostic numbers even when ephemeris conventions are wrong. The key risks are unit mismatches (days/hours/seconds), windowing mistakes, and inconsistent epoch definitions.

## File: `transit/result.py`

### Model: `TransitTime`

- Units:
  - `tc`: days (BTJD expected for TESS).
  - `tc_err`: days.
  - `depth_ppm`: ppm.
  - `duration_hours`: hours.
- Semantics:
  - `epoch`: integer epoch relative to `t0` used by the measuring function.
  - `is_outlier`/`outlier_reason` are purely diagnostic (no policy).

### Model: `TTVResult`

- Units:
  - `o_minus_c`: seconds.
  - `rms_seconds`: seconds.
  - `linear_trend`: seconds per epoch.
- Serialization:
  - `to_dict()` rounds values for stable JSON output (presentation only; underlying floats remain full precision).

### Model: `OddEvenResult`

- Units:
  - depths in ppm; `relative_depth_diff_percent` is a percent (not fraction).

## File: `transit/timing.py`

### Function: `measure_single_transit(...)`

- Units:
  - `time`/`t_center_expected`: days; `duration_hours` converted to days.
- Windowing:
  - Extracts a symmetric window around the expected center (scale set by `duration_days * window_factor`).
- Fit:
  - Fits trapezoid parameters `(tc, depth, duration_days, ingress_ratio)` via L-BFGS-B minimizing chi-squared.
  - Depth initial guess from median in/out-of-window subsets (within the local window).
- Output:
  - Returns `(tc, tc_err, depth_fraction, duration_hours_measured, snr, converged)`.
  - `snr` is computed as `depth / std(residuals)` (not inverse-variance matched; acceptable for triage).

### Function: `measure_all_transit_times(...)`

- Epoch enumeration:
  - Uses data `time_min/time_max` to choose an epoch range around `[min,max]`.
  - For each epoch, measures the transit window and keeps only `converged` transits meeting `min_snr`.
- Unit conversion:
  - Stores `depth_ppm = depth * 1e6`.

### Function: `compute_ttv_statistics(...)`

- O-C:
  - Expected times: `t0 + epoch * period` (days).
  - Residuals: `(observed - expected) * 86400` to seconds.
- Outliers:
  - `_flag_outliers` uses MAD-scaled O-C and (optionally) duration deviation to set `is_outlier`.
- Periodicity:
  - `_compute_periodicity_significance` uses Lomb–Scargle on epoch-number series after removing a linear trend.

## Cross-references

- Public wrapper audit: `working_docs/physics_audit/modules/completed/api_timing.md`
- Odd/even vetting primitives audit: `working_docs/physics_audit/modules/completed/api_vet_and_vetting_primitives.md`

## Fixes / follow-ups

No physics correctness issues identified in these modules.

