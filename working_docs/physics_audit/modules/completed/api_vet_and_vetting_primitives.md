# Module Review: `api/vet.py` and `api/vetting_primitives.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Scope (files / entrypoints)

- `src/bittr_tess_vetter/api/vet.py` → `vet_candidate` (top-level orchestration entrypoint)
- `src/bittr_tess_vetter/api/vetting_primitives.py` → re-exported odd/even primitives:
  - `split_odd_even`
  - `compare_odd_even_depths`
  - `compute_odd_even_result`

## Why this is high leverage

`vet_candidate` is where multiple evidence tiers converge (LC-only, catalog, pixel, exovetter). A unit mismatch or mask convention mismatch at this boundary can silently skew downstream interpretation.

`api/vetting_primitives.py` exports building blocks that host apps may call directly; it must be explicit about assumptions (normalized flux, BTJD days, duration hours) and stable under edge cases.

---

## Function: `vet_candidate`

- Location: `src/bittr_tess_vetter/api/vet.py`
- Public API? yes (`bittr_tess_vetter.api.vet_candidate`)
- Called early by agents? yes (this is the main public vetting entry point)

### Inputs / outputs

- Units + conventions:
  - `lc.time`: days (expected BTJD for TESS)
  - `candidate.ephemeris.period_days`: days
  - `candidate.ephemeris.t0_btjd`: days (same time base as `lc.time`)
  - `candidate.ephemeris.duration_hours`: hours
  - No unit conversion is performed in this orchestrator; it forwards inputs to tier functions.
- Output semantics:
  - Returns `VettingBundleResult(results=[CheckResult...], provenance=..., warnings=[...])`
  - `policy_mode` is forwarded only to LC-only tier (`vet_lc_only`) and is recorded in provenance.

### Physics correctness

- Formula/source: orchestration only (delegates physics to tier modules).
- Assumptions:
  - Light curve is pre-normalized appropriately for LC-only checks (handled in `api/lightcurve.py` and `api/lc_only.py`).
  - Candidate ephemeris uses conventions from `CONVENTIONS.md` (period/t0 days; duration hours).
- Known failure regimes:
  - If caller passes mismatched time bases (`lc.time` not aligned with `t0_btjd`), downstream checks will produce incorrect results without explicit detection at this layer (expected: host provides correct BTJD conventions).

### Numerical stability / data hygiene

- `vet_candidate` itself does not mask NaNs/quality flags; it relies on the tier functions:
  - LC-only tier operates on `LightCurve.valid_mask` (audited in `validation_lc_checks.md`).
  - Pixel tier assumes `TPFStamp` has already been cleaned/masked at creation time.

### Potential orchestration issues (logic, not physics)

- Catalog tier enabling mismatch (fixed):
  - Default enabling now adds V06 only when `network` and `ra_deg+dec_deg` are present, and adds V07 only when `network` and `tic_id` is present.
  - If a caller explicitly enables V06/V07 but required metadata is missing, the orchestrator emits a warning and returns a per-check skipped result (no silent drop).
  - This is evidence-availability policy (not a unit bug), but it materially affects downstream completeness.

### Cross-references (already audited elsewhere)

- LC-only checks: `working_docs/physics_audit/modules/completed/validation_lc_checks.md`
- Exovetter checks: `working_docs/physics_audit/modules/completed/validation_exovetter_checks.md`
- WCS localization: `working_docs/physics_audit/modules/completed/api_wcs_localization.md`
- Pixel PRF stack: `working_docs/physics_audit/modules/completed/api_pixel_prf.md` (when invoked via pixel tier)

---

## File: `api/vetting_primitives.py`

This file is a citation-decorated re-export surface (no numerical modifications). The physics is implemented in:
- `src/bittr_tess_vetter/transit/vetting.py` (odd/even splitting + depth comparison)

### Function: `split_odd_even`

- Location: `src/bittr_tess_vetter/transit/vetting.py` (re-exported in `api/vetting_primitives.py`)
- Public API? yes (via `api/vetting_primitives.py`)
- Units + conventions:
  - `time`: days (BTJD expected)
  - `period`: days
  - `t0`: days
  - `duration_hours`: hours (converted to days via `/24`)
  - Epoch definition matches `validation.lc_checks` parity convention:
    - `epoch = floor((t - t0 + period/2) / period)` (boundaries between transits)
  - In-transit mask uses phase window of half-width `0.75 * duration_days` (total window = 1.5× duration).
- Edge handling:
  - If `duration_days >= 0.5 * period`, returns empty arrays and `n_odd_transits=n_even_transits=0` (defensive: odd/even not meaningful).
  - Does not validate finiteness of `time`/`flux`/`flux_err` inputs; downstream depth computation filters finite values.

### Function: `compare_odd_even_depths`

- Location: `src/bittr_tess_vetter/transit/vetting.py`
- Units + conventions:
  - Assumes flux normalized near 1.0; depth is computed as `1.0 - flux`.
  - Returns depths and differences in **ppm**.
- Statistics:
  - Uses inverse-variance weights from `flux_err`.
  - Returns `diff_err_ppm = sqrt(err_odd^2 + err_even^2) * 1e6`.
  - Significance is `abs(depth_odd - depth_even) / diff_err` (sigma).
- Edge handling:
  - Requires at least 3 valid points per parity; else returns zeros with `diff_err_ppm=inf`.

### Function: `compute_odd_even_result`

- Location: `src/bittr_tess_vetter/transit/vetting.py`
- Units + conventions:
  - Same as above; outputs an `OddEvenResult` with `relative_depth_diff_percent` (percent, not fraction).
  - Relative difference uses average depth as denominator (avoids division-by-zero).
- Relationship to LC-only V01:
  - This primitive uses a simple baseline assumption (baseline=1.0 and in-transit-only depth).
  - LC-only V01 (`validation/lc_checks.check_odd_even_depth`) uses per-epoch local baselines and different aggregation.
  - They agree on epoch/parity definition but may differ numerically if flux normalization drifts or baselines vary.

---

## Follow-ups / potential improvements

1) `vet_candidate` catalog gating: implemented (see note above).
2) `transit/vetting.py` input validation: implemented (`period>0`, `duration_hours>0`, finite `t0` guards).
