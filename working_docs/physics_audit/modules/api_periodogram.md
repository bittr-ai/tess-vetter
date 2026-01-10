# Module Review: `api/periodogram.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is early / high leverage

This is the first “real science” step agents will run to detect periodicity:
- transit discovery (TLS) and/or
- rotation/variability (Lomb–Scargle).

If periodogram semantics are off (units, priors, SNR/FAP meanings), everything downstream is miscalibrated.

## Scope (functions)

- `run_periodogram` / `auto_periodogram` (method routing + defaults)
- TLS search integration (priors, duration grid, detection threshold semantics)
- LS periodogram integration (frequency grid, normalization, FAP semantics)
- `refine_period` (resolution/refinement correctness)
- `compute_bls_model` / `compute_transit_model` re-exports (parameterization sanity)
- Any helper functions that:
  - infer cadence or time span
  - mask/clean data
  - convert between depth fraction and ppm

## Audit checklist (to fill)

### Units + conventions

- [x] `min_period`/`max_period` are days; TLS durations are hours; LS has no duration
- [x] TLS depth is returned as `depth_ppm` (LS has no depth)
- [x] Returned `t0` is in the same time basis as input `time` (BTJD days)

### Statistical semantics

- [ ] LS `fap` is not computed (currently set to 1.0 sentinel); document as “unknown”
- [ ] TLS `snr`/`fap` match TLS conventions (verify against TLS docs)
- [ ] `detection_threshold` meaning is stable across methods

### Numerical stability / edge cases

- [ ] Works on short baselines (few periods)
- [ ] Handles gapped cadence, NaNs, quality masks
- [ ] Avoids pathological period grids (too fine, too coarse)

### Tests

- [x] Existing: wrapper returns finite LS peak power (`tests/test_api/test_periodogram_wrappers.py`)
- [x] Add: LS synthetic sinusoid recovery (`tests/test_api/test_periodogram_wrappers.py::test_run_periodogram_ls_recovers_sinusoid_period`)
- [ ] Add: TLS synthetic transit recovery (lightweight, skip if TLS missing)

## Notes (initial pass)

- LS `t0` is now estimated as the epoch of maximum of a best-fit sinusoid at `best_period`
  (primarily for phase-fold visualization; absolute epoch is not physically meaningful).
