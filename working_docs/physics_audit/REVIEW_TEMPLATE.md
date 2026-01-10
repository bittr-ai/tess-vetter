# Review Template (per function)

Copy this into each module review and fill it out function-by-function.

## Function

- Name:
- Location:
- Public API? (`api.*` export) yes/no
- Called early by agents? yes/no

## Inputs / outputs

- Units + conventions:
- Valid ranges:
- Output semantics (including `passed=None` if metrics-only):

## Physics correctness

- Formula/source:
- Assumptions (e.g., circular orbit, limb darkening model, etc.):
- Known failure regimes:

## Statistics / uncertainty

- Estimator used (median, MAD, LS power, TLS SNR, etc.):
- Uncertainty model:
- Few-point dominance / robustness:

## Numerical stability

- NaN handling:
- Empty-mask handling:
- Sensitivity to cadence/gaps:

## Tests

- Existing tests covering this:
- New tests to add:
- Synthetic cases needed:

## Fixes / changes (if any)

- Proposed fix:
- Backwards-compat impact:
- Rollout notes:

