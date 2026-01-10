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

- [x] LS `fap` is not computed (returned as `None` to mean “unknown / not computed”)
- [x] TLS `snr`/`fap` follow TLS outputs (`results.snr`, `results.FAP` when available)
- [x] No explicit `detection_threshold`; TLS significance is exposed via metrics (SDE/SNR/FAP) and downstream callers choose thresholds

### Numerical stability / edge cases

- [x] Works on short baselines (few periods); TLS returns empty/low-SDE results when baseline is insufficient
- [x] Handles gapped cadence, NaNs, quality masks (finite filtering + sort-by-time in `auto_periodogram`)
- [x] Avoids pathological period grids (LS uses fixed log-spaced grid with enforced `min_period < max_period <= baseline/2`)

### Tests

- [x] Existing: wrapper returns finite LS peak power (`tests/test_api/test_periodogram_wrappers.py`)
- [x] Add: LS synthetic sinusoid recovery (`tests/test_api/test_periodogram_wrappers.py::test_run_periodogram_ls_recovers_sinusoid_period`)
- [x] Add: TLS synthetic transit recovery (lightweight, skip if TLS missing)

## Notes (initial pass)

- LS `t0` is now estimated as the epoch of maximum of a best-fit sinusoid at `best_period`
  (primarily for phase-fold visualization; absolute epoch is not physically meaningful).
- LS `fap` is `None` (not computed); TLS `fap` is `results.FAP` when provided by TLS, else `None`.
