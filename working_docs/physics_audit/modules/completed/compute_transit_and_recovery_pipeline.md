# Module Review: `compute/transit.py` + `recovery/pipeline.py` + `domain/detection.py` (partial)

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is high leverage

- `compute/transit.py` defines the baseline in/out-of-transit masking and “depth + SNR proxy” measurement used across multiple pipelines and tests.
- `recovery/pipeline.py` is the “active-star recovery” core; errors here can create convincing but wrong recovered signals.
- `domain/detection.py` defines candidate/result contracts and enforces key unit constraints (days vs hours, fractional depth).

## File: `domain/detection.py` (contract sanity)

### Model: `TransitCandidate`

- Units + conventions:
  - `period`: days (validated `> 0`)
  - `t0`: BTJD days (not range-validated; treated as consistent with time array)
  - `duration_hours`: hours (validated `> 0`)
  - `depth`: fractional depth (validated `0 < depth <= 1`)
  - `snr`: non-negative float
- Notes:
  - `duration_days` helper converts hours→days via `/24`.

### Model: `PeriodogramPeak` / `PeriodogramResult`

- Units:
  - periods are `PeriodDays` (gt 0), `t0` is BTJD days.
- Guardrails:
  - `PeriodogramResult` enforces `n_periods_searched >= 1` for non-TLS methods.

## File: `compute/transit.py`

### Function: `get_transit_mask`

- Units:
  - `time`, `t0` in days (BTJD), `period` in days, `duration_hours` in hours.
- Method:
  - phase centered on transit: `phase = ((t - t0)/P + 0.5) % 1 - 0.5` in cycles
  - convert duration to phase half-width: `half_duration_phase = duration_days / (2*P)`
  - in transit if `|phase| < half_duration_phase`
- Assumptions:
  - box-shaped “in-transit” window; no ingress/egress modeling.
- Risks:
  - No explicit guards for `period <= 0`; callers generally validate before calling.

### Function: `measure_depth`

- Semantics:
  - depth = `(mean_out - mean_in) / mean_out` (fractional), intended positive for dips.
  - uncertainty via propagation of SEMs of in/out means.
- Numerical stability:
  - raises if no in- or out-of-transit points.
  - uses `ddof=1` when `n>1`, else std=0.
  - if `mean_out == 0`, returns `depth_err = NaN`.
- Known failure regimes:
  - Sensitive to outliers; uses mean, not median/robust estimator.
  - Assumes flux is normalized near 1.0 (true for most API paths).

### Function: `fold_transit`

- Semantics:
  - returns phase in `[-0.5, 0.5]` centered on transit and flux sorted by phase.

### Function: `detect_transit`

- Semantics:
  - uses `get_transit_mask`, then `measure_depth`
  - if measured depth is negative, logs warning and uses absolute value (treats “anti-transit” as a data quality indicator).
  - computes an SNR proxy:
    - `scatter = std(out_of_transit_flux)`
    - `snr = depth * sqrt(n_in_transit_points_total) / scatter` (capped to `MAX_SNR`)
- Risks:
  - SNR proxy ignores time-correlated noise and epoch-to-epoch baseline drift; expected for a cheap proxy.
  - If flux contains trends, `scatter` may be overestimated or underestimated depending on detrending.

### Tests

- Existing tests:
  - `tests/test_compute/test_compute.py` exercises `get_transit_mask`, `measure_depth`, `detect_transit` behaviors and edge cases.
  - `tests/test_integration/test_high_leverage_integration.py` also exercises mask behavior.

## File: `recovery/pipeline.py`

### Function: `recover_transit_timeseries`

- Inputs / units:
  - `time` days, `flux` normalized, `flux_err` fractional
  - `period` days, `t0` days (BTJD), `duration_hours` hours
- Steps:
  1) sorts by time and casts to float64
  2) estimates rotation period if missing (fallback to 5d if SNR < 3)
  3) builds a widened transit mask (`duration_hours * 1.5`) using `get_transit_mask`
  4) detrends using `detrend_for_recovery` (harmonic or wotan methods)
  5) re-normalizes by out-of-transit median
  6) stacks transits and fits trapezoid model in phase space
  7) returns detection boolean based on `depth/depth_err` threshold
- Physics assumptions:
  - detrending model is intended to remove stellar variability without erasing the transit (hence transit masking).
  - trapezoid model is a simplified transit shape; sufficient for recovery/triage, not physical parameter inference.
- Numerical stability:
  - validates equal lengths, minimum points, `period>0`, `duration_hours>0`
  - raises if trapezoid fit fails to converge.
  - handles empty out-of-transit mask by defaulting medians/stds.

### Tests

- Existing tests:
  - `tests/recovery/test_primitives.py` covers the underlying primitives (stacking, rotation estimation, detrending, trapezoid fit).
  - `tests/test_api/test_recovery_api.py` covers multi-sector/time-ordering recovery sanity and input requirements.

## Fixes / follow-ups (non-blocking)

- Consider adding explicit `period > 0` guard inside `get_transit_mask` for clearer errors (currently upstream callers validate).
- Consider a more robust depth estimator (median-based) for `measure_depth` if it becomes used for final metrics rather than proxy scoring.

