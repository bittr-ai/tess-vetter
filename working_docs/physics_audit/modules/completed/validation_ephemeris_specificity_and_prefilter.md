# Module Review: `validation/ephemeris_specificity.py` + `api/ephemeris_specificity.py` + `validation/prefilter.py` + `api/prefilter.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is high leverage

These modules provide cheap “sanity/fragility” diagnostics that host pipelines often use to:
- decide whether a candidate is robust vs alias/systematics-prone
- gate more expensive compute (or pick thresholds)

The biggest physics risks are unit drift (days/hours) and inconsistent masking definitions for “in transit” vs “out of transit”.

## File: `validation/ephemeris_specificity.py`

### Scope

Implements a NumPy mirror of the MLX smooth-template score and related diagnostics:
- smooth template construction (differentiable box)
- phase-shift null tests
- t0 local sensitivity scans
- concentration / few-point dominance measures

### Core conventions

- `time`: days (BTJD expected in TESS context)
- `period_days`: days
- `t0_btjd`: days
- `duration_hours`: hours (converted to days internally)
- `flux`: normalized near 1.0; uses `y = 1 - flux` as “depth-like” signal
- Uses `validation.base.get_in_transit_mask` for in-transit selection in concentration metrics (shared convention with other validation checks).

### Notable behaviors

- `smooth_box_template_numpy` matches the MLX shape: logistic edges controlled by:
  - `ingress_egress_fraction` and `sharpness`
  - floor on ingress timescale of `1e-6` days to avoid division blowups
- `compute_phase_shift_null`:
  - generates alternative `t0` values either grid or random and evaluates null score distribution
  - returns one-sided p-value estimate with +1 smoothing
- `compute_concentration_metrics`:
  - uses absolute contribution weights `|w * template * y|` to compute:
    - fraction of contribution inside the in-transit window
    - max-point dominance fraction
    - top-5 dominance fraction
    - effective number of points via entropy

### Tests

- No dedicated tests found for `validation/ephemeris_specificity.py` yet (it may be exercised indirectly by host tools).

## File: `api/ephemeris_specificity.py`

- Pure re-export facade over `validation.ephemeris_specificity` (no physics logic).
- Provides stable import paths for host apps.

## File: `validation/prefilter.py`

### Function: `compute_depth_over_depth_err_snr`

- Units + conventions:
  - `depth_fractional` is fractional depth (not ppm)
  - calls `validation.base.get_in_transit_mask` and `get_out_of_transit_mask`
  - depth error from `measure_transit_depth`, then returns `|depth| / depth_err`
- Notes:
  - This is a “depth over depth_err” proxy; it is not a full transit SNR model (does not account for red noise).

### Function: `compute_phase_coverage`

- Semantics:
  - folds phase to `[0,1)` and bins into `n_bins`
  - `coverage_fraction` = fraction of phase bins with any data
  - `transit_phase_coverage` = fraction of selected “transit bins” that have any data; default bins `(0, 1, -1)` are near phase 0 (wrap).
- Risks:
  - If `period_days <= 0` or non-finite, phase folding is undefined (no explicit guard here).

### Tests

- Existing tests:
  - `tests/validation/test_prefilter.py` covers `compute_phase_coverage` and `compute_depth_over_depth_err_snr` sanity.

## File: `api/prefilter.py`

- Pure re-export facade over `validation.prefilter`.

## Fixes / follow-ups (non-blocking)

- Add small direct unit tests for `validation/ephemeris_specificity.py` covering:
  - consistency of score under phase wrap
  - behavior for pathological inputs (period <= 0, NaNs)
- Consider adding input validation guards for `period_days > 0` in `prefilter.compute_phase_coverage` to prevent silent NaNs.

