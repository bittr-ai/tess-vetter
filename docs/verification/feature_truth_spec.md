# Feature Truth Spec (btv-owned)

This document defines **what each high-impact feature means**, its **source-of-truth algorithm**, and the **correctness tests** we use to validate behavior.

Goal: correctness for planet triage, not strict parity with any legacy pipeline.

## Scope (initial)

This spec focuses on the three feature families that most strongly affect triage outcomes:

1) **ModShift (secondary/eclipse evidence)** — `V11` (exovetter) + `V11b` (btv independent)
2) **Pixel time-series host discrimination** — pixel windowed PRF fits
3) **Host plausibility (dilution + physical impossibility)** — dilution math + implied companion size

## 1) ModShift (V11, exovetter) — canonical definition

### Fields

- `modshift_secondary_primary_ratio` (**canonical**): the **exovetter V11 ModShift** secondary/primary *signal significance* ratio.
- `modshift_fred`: exovetter ModShift FRED statistic (false-alarm related).
- `modshift_significant_secondary`: btv policy-derived boolean from the V11 ratio (`ratio > 0.5`).

### Definition / source-of-truth

The canonical `modshift_secondary_primary_ratio` is computed from exovetter outputs as:

- `ratio = |sigma_sec| / |sigma_pri|` (with 0.0 fallback when `sigma_pri` is absent or zero)

This is the behavior implemented in:

- `src/bittr_tess_vetter/validation/exovetter_checks.py:run_modshift`

References:

- Thompson et al. 2018, ApJS 235, 38 (Kepler DR25 catalog / Robovetter methodology; ModShift technique lineage).
- Coughlin et al. 2016, ApJS 224, 12 (Robovetter / false positive identification; ModShift).

### Correctness tests

We validate *behavior* (not exact numeric equality to legacy) with:

- A/B synthetic injection: adding an eclipse-like secondary at phase 0.5 should increase `modshift_secondary_primary_ratio` compared to a transit-only light curve.

See:

- `tests/test_correctness/test_modshift_truth.py`

### V11b (btv independent) is a separate ratio

To avoid name collisions, the V11b-derived ratio is:

- `v11b_secondary_primary_ratio = v11b_sig_sec / v11b_sig_pri`

and **must not** be stored under `modshift_secondary_primary_ratio`.

This preserves interpretability: V11 ratio is “the exovetter ModShift ratio”, while V11b ratio is “our TESS-calibrated ModShift-uniqueness ratio”.

## 2) Pixel time-series host discrimination — canonical definition

### Fields

- `pixel_timeseries_verdict`: `ON_TARGET` / `OFF_TARGET` / `AMBIGUOUS` / `NO_EVIDENCE`
- `pixel_timeseries_delta_chi2`: runner-up minus best chi-squared; higher means better separation

### Definition / source-of-truth

Given one or more transit windows extracted from a TPF:

1) Fit a PRF-weighted transit amplitude per host hypothesis per window (WLS).
2) Aggregate evidence across windows to total chi-squared per hypothesis.
3) Select best hypothesis, and declare:
   - `ON_TARGET` if best source contains `"target"` and `delta_chi2 >= margin_threshold`
   - `OFF_TARGET` if best is non-target and `delta_chi2 >= margin_threshold`
   - `AMBIGUOUS` otherwise
   - `NO_EVIDENCE` when no usable windows are available (delta is defined as `0.0`)

This is implemented in:

- `src/bittr_tess_vetter/compute/pixel_timeseries.py`

### Correctness tests

We validate the *decision logic* using synthetic PRF-consistent pixel cubes:

- Inject a transit-like signal consistent with a PRF centered on the target → verdict should be `ON_TARGET`.
- Inject the same signal centered on a background hypothesis → verdict should be `OFF_TARGET`.

See:

- `tests/test_correctness/test_pixel_timeseries_truth.py`

## 3) Host plausibility (dilution + physical impossibility) — canonical definition

### Fields

- `host_requires_resolved_followup`: high-level flag driven by crowding/ambiguity and physics.
- `host_physically_impossible_count`: count of scenarios flagged as physically inconsistent.
- `host_feasible_best_source_id`: “best feasible host” among non-impossible scenarios.

### Definition / source-of-truth

For each host hypothesis with flux fraction `f`:

- `depth_correction_factor = 1/f`
- `true_depth = observed_depth * depth_correction_factor`
- implied companion radius uses the standard approximation `delta ≈ (R_p/R_*)^2` for central transits:
  - `R_p/R_* ≈ sqrt(true_depth_frac)`

We then flag scenarios as inconsistent if implied radius exceeds conservative bounds (e.g., >2 R_Jup planet limit, or clearly stellar companion).

Implementation:

- `src/bittr_tess_vetter/validation/stellar_dilution.py`
- Aggregation for “best feasible host”:
  - choose the non-impossible scenario with **lowest depth correction factor** (largest flux fraction)
  - `src/bittr_tess_vetter/features/aggregates/host.py`

### Correctness tests

We validate:

- dilution math monotonicity (fainter host → larger corrected depth → larger implied radius)
- physical impossibility flags fire when implied radius crosses thresholds
- best-feasible selection ignores impossible scenarios

See:

- `tests/test_correctness/test_host_plausibility_truth.py`

