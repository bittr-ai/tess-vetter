# Module Review: `api/lc_only.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is early / high leverage

These checks are the first “cheap” vetting an agent will run on a candidate ephemeris.
If they’re miscalibrated (units, masks, statistics), they create false confidence early.

## Scope (functions)

- `odd_even_depth`
- `secondary_eclipse`
- `duration_consistency`
- `depth_stability`
- `v_shape`
- `vet_lc_only` (aggregation/policy plumbing, not physics)

## Audit checklist (to fill)

### Units + conventions

- [x] Period in days, t0 in BTJD days, duration in hours (via `Ephemeris` + wrappers)
- [ ] Depth handling consistent: ppm vs fractional depth (input and output)

### Statistical semantics

- [ ] Odd/even statistic definition and uncertainty model is clear
- [ ] Secondary eclipse search window and significance thresholds are clear
- [x] Depth stability (per-transit depth measurement) is robust to outliers (epoch-level outliers are flagged)
- [x] V-shape metric corresponds to an EB-like shape, not limb-darkened planets (trapezoid fit can distinguish triangle-like vs box-like)

### Edge cases

- [ ] Few transits / few points per transit
- [ ] Gapped cadence, NaNs, masked cadences
- [ ] Long duration relative to period

### Tests

- [x] Synthetic: clean injected transit → odd/even consistent (not suspicious when sufficient data)
- [x] Synthetic: alternating depths → odd/even mismatch detected (`tests/test_api/test_lc_only.py`)
- [x] Synthetic: secondary at phase 0.5 detected when injected (`tests/test_api/test_lc_only.py`)
- [x] Duration-density scaling behaves as expected (dense star shorter, giant longer) (`tests/test_api/test_lc_only.py`)
- [x] Depth stability flags outlier epochs (`tests/test_api/test_lc_only.py`)
- [x] V-shape distinguishes triangle-like vs box-like (`tests/test_api/test_lc_only.py`)
