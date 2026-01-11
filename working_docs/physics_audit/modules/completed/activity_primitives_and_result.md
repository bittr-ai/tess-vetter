# Module Review: `activity/primitives.py` + `activity/result.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is high leverage

Activity characterization influences how we search for and vet transits in difficult regimes (rotators, flare stars). The main physics risks are:
- unit mistakes (days vs hours/minutes, ppm vs fraction),
- time-base assumptions (BTJD consistency),
- cadence/gap handling that can silently break on real light curves.

## File: `activity/result.py`

### Model: `Flare`

- Units:
  - `start_time/end_time/peak_time`: days (BTJD expected when used with TESS).
  - `amplitude`: fractional flux increase above baseline.
  - `duration_minutes`: minutes.
  - `energy_estimate`: ergs (very rough, order-of-magnitude only).
- Serialization:
  - `to_dict()` rounds time to 1e-6 days and duration to 0.01 minutes.

### Model: `ActivityResult`

- Units:
  - `rotation_period`, `rotation_err`: days.
  - `variability_ppm`: ppm.
  - `flare_rate`: flares/day.
  - `activity_index`: unitless 0–1 proxy.
- Output semantics:
  - `to_dict()` is presentation-only rounding; raw values remain unrounded in the dataclass.

## File: `activity/primitives.py`

### Function: `detect_flares(time, flux, flux_err, ...)`

- Units:
  - `time`: days; `baseline_window_hours`: hours; `min_duration_minutes`: minutes.
  - Outputs `Flare` objects with times in days and duration in minutes.
- Method:
  - Builds a rolling median baseline and rolling MAD scatter estimate over a time-derived window.
  - Flags points with residuals above `sigma_threshold * local_scatter`, then groups nearby detections.
- Gap/cadence handling:
  - Cadence is estimated from median of positive `diff(time)`, preferring diffs `< 0.5` days (treating larger gaps as breaks).
  - If all diffs are “gaps”, falls back to overall median cadence (prevents median-of-empty failures).

### Function: `measure_rotation_period(time, flux, ...)`

- Units:
  - periods in days; outputs `(period_days, period_err_days, snr)`.
- Method:
  - Lomb–Scargle with log-spaced period grid; SNR via MAD.
  - Period uncertainty from peak width at half-maximum.
- Guardrail:
  - If flux has zero variance, returns `(1.0, 1.0, 0.0)` (no detection) to avoid Lomb–Scargle normalization issues.

### Function: `classify_variability(periodogram_power, phase_amplitude, flare_count, baseline_days)`

- Units:
  - `phase_amplitude` is fractional; internally converted to ppm for thresholds.
  - `baseline_days` used to compute flare rate (flares/day).
- Classification:
  - Priority order: flare_star → quiet → spotted_rotator → quiet fallback.

### Function: `compute_activity_index(variability_ppm, rotation_period, flare_rate)`

- Units:
  - variability in ppm; rotation in days; flare rate in flares/day.
- Output:
  - 0–1 proxy combining log-scaled variability, rotation, and flare rate.

### Function: `mask_flares(time, flux, flares, buffer_minutes=...)`

- Units:
  - buffer in minutes; converts to days for time masking.
- Method:
  - Replaces flare-region points with linear interpolation between neighboring baseline points.

### Function: `compute_phase_amplitude(time, flux, period, n_bins=...)`

- Units:
  - `period` is in days.
- Note:
  - Uses `time % period` for phase folding, which is numerically safest when `time` is BTJD-like (order ~1e3 days), as in TESS.

### Function: `generate_recommendation(...)`

- Output:
  - Returns a human-readable recommendation and a `recover_transit`-style parameter dict.
  - `min_detectable_depth_ppm` scales roughly as `5 * residual_scatter_ppm / sqrt(n_expected_transits)`.

## Fixes / follow-ups

- Fixed cadence estimation edge case in `detect_flares` (avoid median-of-empty when all diffs are treated as gaps).
- Added a no-variance guard for `measure_rotation_period` to avoid spurious Lomb–Scargle behavior on constant flux.

