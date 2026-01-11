# Module Review: `api/mlx.py` + `compute/mlx_detection.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is high leverage

These modules provide the “MLX path” for fast, differentiable, template-based scoring (and attribution-style diagnostics) that multiple downstream tools treat as **detectability/robustness evidence**.

Key risks:
- silent CPU fallback (or total loss of diagnostics) if MLX optionality is brittle
- unit mismatches (days vs hours, BTJD vs other) that yield plausible-but-wrong scores
- overconfidence if users interpret the score as a calibrated planet probability

## File: `api/mlx.py`

### Function(s)

- Name: `smooth_box_template`, `score_fixed_period`, `score_fixed_period_refine_t0`, `score_top_k_periods`, `integrated_gradients`
- Location: `src/bittr_tess_vetter/api/mlx.py`
- Public API? yes (stable facade surface for host apps)
- Called early by agents? yes (often used for “quick score / null tests / attribution” in host pipelines)

### Inputs / outputs

- Units + conventions:
  - `time`: expected in **days** (BTJD for TESS use cases)
  - `period_days`: **days**
  - `t0_btjd`: **days**
  - `duration_hours`: **hours**
  - `flux`: normalized with baseline ~1.0
  - `flux_err`: same units as `flux` (fractional); optional
- Output semantics:
  - Facade forwards directly to `compute.mlx_detection` functions and returns MLX-native arrays/objects.

### Optional dependency semantics (MLX)

- The facade defines `MLX_AVAILABLE` via `find_spec("mlx")`.
- Importing `api/mlx.py` imports `bittr_tess_vetter.compute.mlx_detection`, which does **not** import MLX eagerly; it imports MLX only inside `_require_mlx()` called by each function.
- Therefore the “MLX is optional” claim is currently correct: importing the facade should not require MLX; calling functions will raise an actionable `ImportError` if MLX is missing.

## File: `compute/mlx_detection.py`

### Function: `_require_mlx`

- Purpose: runtime gate for optional dependency.
- Behavior: imports `mlx.core as mx` inside function and raises an actionable `ImportError` on failure.

### Function: `smooth_box_template`

- Units + conventions:
  - phase is computed in **days** and wrapped into `[-0.5, 0.5)` via `phase - floor(phase + 0.5)`
  - `duration_hours` converted to days via `/24`
- Physics correctness:
  - Template is a smoothed “inside-transit” indicator: `sigmoid(k * (half_duration - |dt|))`
  - `duration_hours` is treated as total first-to-fourth contact duration.
  - `ingress_egress_fraction` sets smoothing width as a fraction of duration (bounded below by `1e-6` days).
- Known failure regimes:
  - If `period_days <= 0` or non-finite, phase math becomes invalid (no explicit guard here).
  - If `duration_hours` is extremely small, the logistic becomes very sharp; `k` is clipped indirectly by `ingress >= 1e-6 days`.

### Function: `score_fixed_period`

- Model/estimator:
  - Matched-filter / BLS-family statistic: estimate depth by weighted least squares against the template, then return `depth_hat / depth_sigma`.
  - Assumes flux baseline ≈ 1.0 and “transit depth” is positive in `y = 1 - flux`.
- Uncertainty model:
  - Uses inverse-variance weights `w = 1 / max(flux_err^2, eps)`; if `flux_err is None`, uses uniform weights.
- Numerical stability:
  - Adds `eps` to denominators to avoid divide-by-zero.
  - No explicit NaN masking; callers should pre-mask/cadence-clean.

### Function: `score_top_k_periods`

- Physics correctness:
  - Scores a small set of periods, then returns deterministic softmax weights.
  - These weights are *not* a calibrated probability; they are a convenience weighting over candidate periods.
- Numerical stability:
  - `temperature` bounded below by `1e-6` to avoid division blowups.

### Function: `integrated_gradients`

- Semantics:
  - Standard integrated gradients attribution over `steps` interpolation points from `baseline` to `flux`.
- Numerical considerations:
  - Uses a Python loop over `alphas.tolist()` which is explicit and stable at small `steps`, but can be slow for large steps.

### Function: `score_fixed_period_refine_t0`

- Semantics:
  - Scans a local window around `t0` (default half-span tied to `duration_hours`, capped at 120 minutes) and picks the best score.
- Units:
  - Scan window specified in minutes; converted to days.
- Valid ranges:
  - Enforces `t0_scan_n >= 21` and odd grid size.
- Numerical stability:
  - Uses batched scoring to reduce overhead; forces evaluation via `mx.eval(scores_mx)` then converts to NumPy.

## Tests

- Existing tests covering this:
  - `tests/test_compute/test_mlx_detection.py` (skips if MLX not installed) covers:
    - template range/periodicity/sharpness
    - scoring behavior and t0 refinement
    - integrated gradients interface
- New tests to add (optional):
  - Validate behavior for invalid inputs (non-finite / `period_days <= 0`) returns a clear exception rather than NaNs.
  - Add a “CPU fallback parity” test vs `api/mlx_numpy.py` (if/when a deterministic NumPy mirror is maintained).

## Fixes / changes (if any)

No code changes proposed in this audit pass.

Follow-up candidates:
- add explicit input validation guards for `period_days > 0`, finite `t0`, `duration_hours > 0`
- document that `score_fixed_period` is a matched-filter SNR proxy, not a probability

