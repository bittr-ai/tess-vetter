# FPP Binning Spec

## Goal
Add first-class light-curve **binning** as an alternative to current point-cap/downsampling before `TRICERATOPS calc_probs()`, so agents can choose a reduction strategy explicitly and reproducibly.

## Motivation
Current `max_points` uses index downsampling (subset selection). Upstream TRICERATOPS tutorial often bins to ~100 points for speed. For shallow/noisy transits, binning may produce more stable FPP behavior than raw-point subsampling.

## Non-goals
- No silent runtime policy mutation based on timeout.
- No automatic strategy switching in this phase.
- No backward-compat guarantees for deprecated v3 policy internals.

## User-facing Contract

### New runtime knobs
- `--point-reduction [downsample|bin|none]`
- `--target-points INTEGER` (new canonical field)
- `--bin-stat [mean|median]` (default `mean`)
- `--bin-err [propagate|robust]` (default `propagate`)

### Migration input alias
- Accept legacy `--max-points` as input alias during migration.
- Canonical output/provenance field is `target_points`.

### Semantics
- `downsample`: keep approximately `target_points` evenly-spaced original folded/windowed points.
- `bin`: aggregate folded/windowed light curve into approximately `target_points` uniform time bins.
- `none`: keep all windowed points (ignore `target_points`).

### Validation
- `target_points >= 20` when reduction is `bin` or `downsample`.
- Reject incompatible combinations (`point_reduction=none` + explicit `target_points`) in strict mode; otherwise ignore `target_points` with warning.

## Pipeline Stage (Important)
Point reduction occurs **after phase-folding and transit-window selection** and before `target.calc_probs(...)`.
- Binning/downsampling must operate in `time_folded` space, not raw BTJD timeline.

## API/Schema Changes

### Effective runtime policy
Add fields:
- `point_reduction`
- `target_points`
- `bin_stat`
- `bin_err`

### Result provenance
Add under `fpp_result.effective_config` and `requested_config`:
- `point_reduction`
- `target_points`
- `bin_stat`
- `bin_err`
- `n_points_raw`
- `n_points_windowed`
- `n_points_used`

### Trace
`resolution_trace` must include `point_reduction` and `target_points` sources.

## Implementation Design

### In `validation/triceratops_fpp.py`
Replace current point-cap section with strategy dispatch:

1. Compute `time_folded/flux/flux_err` and window selection (unchanged).
2. Apply reduction strategy:
   - `downsample`: existing `np.linspace(...).astype(int)` path.
   - `bin`:
     - Construct uniform bins over `[min(time_folded), max(time_folded)]`.
     - Aggregate flux by `bin_stat`.
     - Aggregate per-bin uncertainty by `bin_err`:
       - `propagate`: standard error of mean under independent errors,
         `sigma_bin = sqrt(sum(sigma_i^2)) / n_bin`.
       - `robust`: `sigma_bin = 1.4826 * MAD(flux_bin)`.
     - Drop empty bins.
     - Minimum bin occupancy: `n_bin >= 1`.
   - `none`: no reduction.
3. Sort by folded time and pass reduced arrays to `calc_probs`.

### TRICERATOPS uncertainty nuance
TRICERATOPS `calc_probs` takes scalar `flux_err_0`.
- Even if per-bin errors are computed, wrapper currently reduces to one representative scalar.
- Keep this explicit in implementation/provenance so users understand the collapse step.

## Determinism
- Binning must be deterministic for same input arrays and settings.
- No RNG in reduction path.

## Defaults
- `point_reduction=downsample`
- `target_points=1500`
- `bin_stat=mean`
- `bin_err=propagate`

## Practical Guidance
- `bin` is most meaningfully different from `downsample` at lower `target_points` (roughly `100–500`).
- At very high `target_points` near `n_points_windowed`, `bin` converges toward near-identity behavior.

## Performance Expectations
- `bin` should reduce variance and runtime similarly to downsampling at equal target points.
- No timeout-driven capping of `mc_draws` or points.

## Tests

### Unit tests
- Reduction mode selection and validation.
- Binning output sizes and monotonic folded-time order.
- Error propagation behavior sanity checks.
- Determinism test with fixed inputs.
- Bin-count floor case: `target_points > n_points_windowed` clamps gracefully.

### Integration tests
- Existing FPP contract tests still pass.
- New test: same input + same seed, `bin` vs `downsample` produce distinct but valid outputs.
- New test: provenance fields present and correct for each reduction mode.

### Regression tests
- Ensure requested `mc_draws` is honored regardless of timeout.
- Ensure no silent fallback to alternate reduction mode.
- Ensure canonical output uses `target_points` (not legacy `max_points`).

## Rollout Plan
1. Implement reduction strategy in `calculate_fpp_handler`.
2. Wire CLI policy and provenance fields.
3. Update docs/help text and `cli_contracts.md`.
4. Add tests and run full suite.
5. Run TOI-5739 comparison matrix:
   - modes: `bin`, `downsample`
   - target points: `100, 250, 500, 1000, 1500`
   - fixed `mc_draws` and seeds
   - compare disposition stability + FPP spread.

## Acceptance Criteria
- Agents can explicitly choose binning with clear help text.
- Outputs fully disclose reduction strategy and resulting point counts.
- No hidden point/draw caps from timeout logic.
- Full CI and release checks pass.
