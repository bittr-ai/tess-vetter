# Module Review: `compute/model_competition.py` + `api/model_competition.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is high leverage

Model competition is used to detect “convincing false positives”:
- stellar variability that a transit model can partially fit,
- EB-like odd/even depth patterns and secondary eclipses,
- known systematic/alias periods.

The biggest physics risks are unit mismatches (days vs hours, ppm vs fraction), inconsistent epoch parity conventions, and likelihood calculations that behave badly when `flux_err` is missing/zero.

## File: `compute/model_competition.py`

### Units + conventions

- `time`: days (BTJD expected)
- `period`: days
- `t0`: BTJD days
- `duration_hours`: hours (`/24` conversions inside templates)
- `flux`: assumed normalized baseline ~1.0
- depth parameters are **fractional**, with derived `*_ppm` stored as `depth * 1e6`.

### Likelihood + information criteria

- `_compute_log_likelihood` implements a Gaussian log-likelihood on residuals:
  - guards against zero/NaN `flux_err` by flooring variance to `1e-20`.
- `_compute_aic_bic` uses:
  - `AIC = -2 logL + 2k`
  - `BIC = -2 logL + k ln(n)`

### Templates

- `_box_transit_template` returns a 0/1 mask for in-transit points using a centered phase convention:
  - `phase = (t - t0)/P - floor((t - t0)/P + 0.5)` (in `[-0.5,0.5]`)
  - in-transit if `|phase * P| < duration_days/2`

### Fits

- `fit_transit_only`:
  - fits depth via weighted least squares on `1 - flux` against the transit template.
- `fit_transit_sinusoid`:
  - fits depth + sin/cos coefficients at `k/P` for `k=1..n_harmonics` (linear LS).
  - Assumes normalized flux baseline (~1.0); there is no independent intercept term.
- `fit_eb_like`:
  - fits `depth_odd`, `depth_even`, and a phase-0.5 secondary depth via LS.
  - Odd/even parity is based on `floor((t-t0)/P + 0.5)` (consistent with the project’s odd/even epoch convention).

### Model selection

- `run_model_competition` selects the lowest-BIC model and emits:
  - label: `TRANSIT` / `SINUSOID` / `EB_LIKE` / `AMBIGUOUS` depending on `delta_BIC` vs `bic_threshold`.
  - `artifact_risk`: heuristic mapping (0, 0.8, 0.9, or 0.5 when ambiguous).

### Artifact priors

- `check_period_alias` checks proximity to known systematic periods using fractional tolerance.
- `compute_artifact_prior` combines alias risk + simple quality-flag risk into a weighted prior.

## File: `api/model_competition.py`

- Pure re-export API surface over `compute/model_competition.py` (no behavior changes).

## Tests

- `tests/test_compute/test_model_competition.py` covers:
  - each model’s basic behavior on synthetic signals,
  - model selection outcomes,
  - period alias/prior utilities.

## Fixes / follow-ups

- Fixed `_compute_log_likelihood` to be robust when `flux_err` is zero/NaN (common when uncertainties are unavailable).

