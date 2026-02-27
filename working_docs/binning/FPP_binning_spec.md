# FPP Binning Spec

## Goal
Add first-class light-curve **binning** while continuing to support existing **downsampling** and `none` modes before `TRICERATOPS calc_probs()`, with clear migration behavior for legacy `max_points` inputs.

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
- Map `--max-points` to canonical `target_points` while preserving legacy behavior (`point_reduction=downsample`) unless `--point-reduction` is explicitly set.
- Canonical output/provenance field is `target_points`.
- `downsample` remains a supported canonical mode during this phase.
- If both `--max-points` and `--target-points` are supplied:
  - if equal: accept and emit a deprecation warning for `--max-points`;
  - if different: fail with actionable error and require a single source of truth.
- Resolution trace precedence:
  - canonical source for emitted `target_points` is always `target_points` when provided;
  - if equal `max_points` is also provided, record it as a matched legacy alias input in trace metadata.
  - if only `max_points` is provided, emit canonical `target_points` with source `legacy_max_points_alias`.
- Trace metadata schema:
  - `resolution_trace.target_points.source`: one of
    `target_points`,
    `legacy_max_points_alias`,
    `preset_default`,
    `target_points_ignored_for_none`,
    `legacy_max_points_alias_ignored_for_none`.
  - `resolution_trace.target_points.legacy_alias_matched`: boolean.
  - `resolution_trace.target_points.legacy_alias_value`: integer|null.
  - when `point_reduction=none` and a target input was provided then ignored, set:
    - `resolution_trace.target_points.source = target_points_ignored_for_none` (or `legacy_max_points_alias_ignored_for_none` when alias-only input),
    - keep `legacy_alias_*` fields populated as applicable.

### Semantics
- `downsample`: keep approximately `target_points` evenly-spaced original folded/windowed points.
- `bin`: aggregate folded/windowed light curve into approximately `target_points` uniform time bins.
- `none`: keep all windowed points (ignore `target_points`).

### Validation
- `target_points >= 20` when `point_reduction=downsample`.
- `target_points >= 20` when `point_reduction=bin`.
- For `point_reduction=none`, always ignore explicit `target_points` with a warning (no strict-mode branch).
- For `point_reduction=none`, explicit `max_points` is canonicalized to `target_points`, then warn-and-ignore (same behavior as explicit `target_points`).
- Disallow `bin_stat=median` with `bin_err=propagate` (fail with actionable error); use `bin_err=robust` for median binning.

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
Add under `fpp_result.requested_config`:
- `point_reduction`
- `target_points`
- `bin_stat`
- `bin_err`
- `max_points` (optional, legacy input only when provided)

Add under `fpp_result.effective_config`:
- `point_reduction`
- `target_points`
- `bin_stat`
- `bin_err`
- `target_points_clamped` (boolean)
- For `point_reduction=none`, set:
  - `effective_config.target_points = null`
  - `effective_config.target_points_clamped = false`

Add under `fpp_result.runtime_metrics`:
- `n_points_raw`
- `n_points_windowed`
- `n_points_used`
- `flux_err_0`
- `flux_err_0_method`
- `flux_err_0_source_count`
- `bin_err_robust_fallback_bins`
- `low_window_point_count` (boolean)
- `windowed_points_empty` (boolean)
- Non-applicable/default semantics:
  - `bin_err_robust_fallback_bins = 0` when robust fallback is not used.
  - `low_window_point_count = false` unless `1 <= n_points_windowed < 20`.
  - `windowed_points_empty = true` iff `n_points_windowed == 0`.

### Trace
`resolution_trace` must include `point_reduction` and `target_points` sources.

## Implementation Design

### In `validation/triceratops_fpp.py`
Replace current point-cap section with strategy dispatch:

1. Compute `time_folded/flux/flux_err` and window selection (unchanged).
2. Apply reduction strategy:
   - `downsample`: existing `np.linspace(...).astype(int)` path.
   - `bin`:
     - Construct uniform bins over `[min(time_folded), max(time_folded)]` with deterministic edge policy:
       - half-open bins `[left, right)` for all interior bins;
       - last bin closed `[last_left, last_right]`.
     - Emit reduced bin-time coordinate as bin center:
       `time_folded_used_bin = 0.5 * (left_edge + right_edge)`.
     - Aggregate flux by `bin_stat`.
     - Aggregate per-bin uncertainty by `bin_err`:
       - `propagate`: standard error of mean under independent errors,
         `sigma_bin = sqrt(sum(sigma_i^2)) / n_bin`.
       - `robust`: `sigma_bin = 1.4826 * MAD(flux_bin)`.
     - Drop empty bins.
     - Minimum bin occupancy: `n_bin >= 1`.
     - For `bin_err=robust` and `n_bin < 2`, fall back to `propagate` for that bin (and record fallback count in provenance/debug payload).
   - `none`: no reduction.
3. Sort by folded time and pass reduced arrays to `calc_probs`.
4. Clamp behavior:
   - `effective_target_points = min(requested_target_points, n_points_windowed)` for `downsample`/`bin`;
   - set `target_points_clamped=true` when clamped;
   - do not fail for `target_points > n_points_windowed`.
5. Empty-window behavior:
   - if `n_points_windowed == 0`, fail fast with actionable error (`no_windowed_points`) before reduction for all modes (`downsample`, `bin`, `none`).
6. Bin output size tolerance:
   - For `point_reduction=bin` and non-empty windowed inputs, require `n_points_used` in `[1, effective_target_points]`.
   - Treat `n_points_used < effective_target_points` as expected when empty bins are dropped.
7. Sparse-window behavior:
   - if `1 <= n_points_windowed < 20`, continue with available points (no hard failure), set `low_window_point_count=true`, and emit a warning.

### TRICERATOPS uncertainty nuance
TRICERATOPS `calc_probs` takes scalar `flux_err_0`.
- Even if per-bin errors are computed, wrapper reduces to one representative scalar.
- Canonical collapse rule: `flux_err_0 = nanmedian(reduced_flux_err)`.
- Record `flux_err_0_method="nanmedian_reduced_flux_err"` and `flux_err_0_source_count`.

## Determinism
- Binning must be deterministic for same input arrays and settings.
- No RNG in reduction path.

## Defaults
- `point_reduction=downsample` for `fast` preset (legacy-compatible default).
- `point_reduction=none` for `standard` preset.
- `target_points=1500` for `fast` preset.
- `bin_stat=mean`
- `bin_err=propagate`

## Practical Guidance
- `bin` is most meaningful at lower `target_points` (roughly `100–500`).
- At very high `target_points` near `n_points_windowed`, `bin` converges toward near-identity behavior.

## Performance Expectations
- `bin` should reduce variance and runtime similarly to downsampling at equal target points.
- No timeout-driven capping of `mc_draws` or points.

## Tests

### Unit tests
- Reduction mode selection and validation.
- Validation rejection for `bin_stat=median` + `bin_err=propagate`.
- Binning output sizes and monotonic folded-time order.
- Deterministic edge-assignment behavior for points exactly on bin boundaries.
- Error propagation behavior sanity checks.
- Robust-error low-occupancy fallback behavior (`n_bin < 2`).
- Determinism test with fixed inputs.
- Bin-count floor case: `target_points > n_points_windowed` clamps gracefully.
- Sparse-window case (`n_points_windowed < 20`) continues with warning and provenance flag.

### Integration tests
- Existing FPP contract tests still pass.
- New synthetic-fixture test: construct a deterministic fixture where `bin` and `downsample` are guaranteed to produce different `time_folded_used` vectors at `target_points=100`, and assert both runs remain valid.
- New test: provenance fields present and correct for each reduction mode.

### Regression tests
- Ensure requested `mc_draws` is honored regardless of timeout.
- Ensure no timeout-based mutation of requested `mc_draws`.
- Ensure canonical output uses `target_points` (not legacy `max_points`).
- Ensure legacy `max_points` input is accepted with deprecation warnings and canonicalized to `target_points` while preserving `downsample` behavior unless `point_reduction` is explicitly set.
- Ensure `--max-points` + `--target-points` precedence contract (equal: warn; different: fail) is enforced.
- Ensure resolution trace uses canonical `target_points` source precedence and records matched legacy alias input.
- Ensure alias-only input (`max_points` without `target_points`) emits canonical `target_points` with source `legacy_max_points_alias`.
- Ensure trace metadata schema keys are present and populated (`source`, `legacy_alias_matched`, `legacy_alias_value`).
- Ensure `point_reduction=none` + explicit `target_points` is warn-and-ignore.
- Ensure `point_reduction=none` + explicit `max_points` follows the same warn-and-ignore behavior after canonicalization.
- Ensure `point_reduction=none` trace source is `target_points_ignored_for_none` (or `legacy_max_points_alias_ignored_for_none` for alias-only inputs).
- Ensure `point_reduction=none` emits `effective_config.target_points=null` and `target_points_clamped=false`.
- Ensure empty-window (`n_points_windowed == 0`) for any mode fails with `no_windowed_points`.
- Ensure reduced bin time coordinate uses bin centers.
- Ensure default metric semantics (`bin_err_robust_fallback_bins=0`, `low_window_point_count=false` unless sparse-window, `windowed_points_empty` consistency).
- Ensure `bin_stat=median` + `bin_err=propagate` is rejected with actionable error.
- Ensure `flux_err_0` collapse method/provenance fields are present and correct.

## Rollout Plan
1. Implement reduction strategy in `calculate_fpp_handler`.
2. Wire CLI policy and provenance fields.
3. Update docs/help text and `cli_contracts.md`.
4. Add tests and run full suite.
5. Run TOI-5739 comparison matrix:
   - modes: `downsample`, `bin`
   - target points: `100, 250, 500, 1000, 1500`
   - fixed `mc_draws`
   - fixed seeds: `13, 29, 41, 67, 101`
   - compare disposition stability + FPP spread.

## Acceptance Criteria
- CLI help must include executable examples for all reduction modes (`downsample`, `bin`, `none`) and alias migration notes (`max_points` -> `target_points`).
- Outputs must include required provenance fields and runtime metrics specified in this document.
- No hidden point/draw caps from timeout logic.
- On TOI-5739 matrix, median runtime for `bin` at each target-point setting is no worse than 1.5x `downsample`, measured on a single dedicated host class with:
  - warm cache run discarded once per mode/target-point pair;
  - 3 timing repeats per pair;
  - median of repeats used for comparison.
- On TOI-5739 matrix, disposition agreement (`same final class label`) between `bin` and `downsample` is >= 90% across 25 paired runs (5 target-point settings x 5 fixed seeds).
- Full CI and release checks pass.
