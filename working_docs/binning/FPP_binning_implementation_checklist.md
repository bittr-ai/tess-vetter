# FPP Binning Implementation Checklist

## Scope
Implement `point_reduction` with full support for `downsample`, `bin`, and `none` modes for FPP preprocessing, plus explicit legacy alias handling, with full CLI/config/provenance/test coverage.

## Branch
- Target branch: `main` (or `feature/fpp-binning` if desired)

## Team Topology
- `Agent-A` (Core runtime): `validation/triceratops_fpp.py`
- `Agent-B` (CLI/policy/provenance): `cli/fpp_cli.py` + command help
- `Agent-C` (Tests/contracts/docs): `tests/*`, docs
- `Agent-D` (Reviewer): full code review + usability check

## Round 0: Lock Requirements
- [ ] Confirm field names: `point_reduction`, `target_points`, `bin_stat`, `bin_err`.
- [ ] Confirm preset defaults:
  - `fast`: `point_reduction=downsample`, `target_points=1500`
  - `standard`: `point_reduction=none`
- [ ] Confirm no timeout-driven mutation of `mc_draws` or point-reduction knobs; remove existing timeout-based draw clamping logic.
- [ ] Confirm migration rule: accept `max_points` input alias and canonicalize to `target_points` while preserving `point_reduction=downsample` behavior unless `point_reduction` is explicitly provided.
- [ ] Confirm precedence rule for `max_points` + `target_points`:
  - equal values: accept + deprecation warning for `max_points`
  - different values: fail with actionable error
- [ ] Confirm trace precedence rule:
  - emitted `target_points` source is canonical `target_points` when provided
  - matched equal `max_points` is recorded as legacy alias metadata in trace
  - alias-only `max_points` input emits canonical `target_points` source=`legacy_max_points_alias`
- [ ] Confirm trace metadata schema:
  - `resolution_trace.target_points.source`
  - `resolution_trace.target_points.legacy_alias_matched`
  - `resolution_trace.target_points.legacy_alias_value`
- [ ] Confirm `none`-mode ignored-input trace sources:
  - `target_points_ignored_for_none`
  - `legacy_max_points_alias_ignored_for_none`
- [ ] Confirm `point_reduction=none` + explicit `target_points` is warn-and-ignore (not error).
- [ ] Confirm `point_reduction=none` + explicit `max_points` canonicalizes then warn-and-ignore.
- [ ] Confirm `bin_stat=median` + `bin_err=propagate` is rejected with actionable error.

Deliverable:
- [ ] Updated spec if anything changes:
  - `/Users/collier/projects/apps/bittr-tess-vetter/working_docs/binning/FPP_binning_spec.md`

## Round 1: Core Runtime (Agent-A)
Files:
- `/Users/collier/projects/apps/bittr-tess-vetter/src/tess_vetter/validation/triceratops_fpp.py`

Tasks:
- [ ] Add strategy dispatch at folded/windowed stage:
  - `downsample`, `bin`, `none`
- [ ] Implement deterministic folded-time uniform binning.
- [ ] Implement `bin_stat` (`mean`/`median`).
- [ ] Implement `bin_err`:
  - `propagate`: `sqrt(sum(sigma_i^2))/n_bin`
  - `robust`: `1.4826 * MAD`
- [ ] Define/handle empty bins (`drop`) and minimum occupancy (`n_bin >= 1`).
- [ ] Add robust low-occupancy fallback: when `bin_err=robust` and `n_bin < 2`, use `propagate` for that bin and record fallback count.
- [ ] Preserve scalar `flux_err_0` behavior explicitly (representative value from reduced points).
- [ ] Define `flux_err_0` collapse rule as `nanmedian(reduced_flux_err)` and emit provenance fields for method/source count.
- [ ] Ensure `n_points_raw/windowed/used` are accurate.
- [ ] Emit `runtime_metrics.bin_err_robust_fallback_bins`.
- [ ] Emit `runtime_metrics.windowed_points_empty`.
- [ ] Enforce runtime-metric defaults:
  - `bin_err_robust_fallback_bins=0` when unused
  - `low_window_point_count=false` unless `1 <= n_points_windowed < 20`
  - `windowed_points_empty=true` iff `n_points_windowed == 0`
- [ ] Implement deterministic bin-edge policy:
  - interior bins `[left, right)`, final bin `[left, right]`.
- [ ] Emit reduced `time_folded_used` at bin centers (`0.5 * (left_edge + right_edge)`).
- [ ] Implement clamp behavior:
  - `effective_target_points = min(requested_target_points, n_points_windowed)` for `downsample`/`bin`;
  - set `target_points_clamped` provenance flag when clamped.
- [ ] Implement empty-window guard:
  - if `n_points_windowed == 0`, fail fast with `no_windowed_points` for all modes (`downsample|bin|none`).
- [ ] Implement sparse-window behavior:
  - if `1 <= n_points_windowed < 20`, continue with available points, emit warning, set `runtime_metrics.low_window_point_count=true`.
- [ ] Enforce bin output-size tolerance:
  - for `bin`, require `1 <= n_points_used <= effective_target_points`;
  - accept `n_points_used < effective_target_points` when empty bins are dropped.
- [ ] Remove timeout-based mutation of `mc_draws`; always honor requested draws unless explicit validation rejects input.

Acceptance:
- [ ] `calc_probs` receives arrays matching chosen strategy.
- [ ] No timeout logic alters requested draws/points.

## Round 2: CLI/Policy Wiring (Agent-B)
Files:
- `/Users/collier/projects/apps/bittr-tess-vetter/src/tess_vetter/cli/fpp_cli.py`
- `/Users/collier/projects/apps/bittr-tess-vetter/src/tess_vetter/cli/enrich_cli.py` (if needed)

Tasks:
- [ ] Add CLI options:
  - `--point-reduction [downsample|bin|none]`
  - `--target-points`
  - `--bin-stat [mean|median]`
  - `--bin-err [propagate|robust]`
- [ ] Add input alias `--max-points` -> `target_points` with deprecation warning.
- [ ] Extend runtime policy resolution + conflict handling.
- [ ] Enforce precedence rule for `--max-points` + `--target-points` (equal warn, different fail).
- [ ] Enforce trace precedence metadata for equal alias/canonical inputs.
- [ ] Thread policy into runtime call.
- [ ] Update `--help` with concise examples.
- [ ] Ensure `--help` includes executable examples for `downsample|bin|none` and alias migration notes (`max_points` -> `target_points`).

Acceptance:
- [ ] `resolution_trace` includes reduction fields.
- [ ] Conflicts fail only where specified (`--max-points` and `--target-points` mismatch); `point_reduction=none` + `target_points` warns and ignores.
- [ ] `bin_stat=median` + `bin_err=propagate` fails with actionable error.
- [ ] `point_reduction=none` emits `effective_config.target_points=null` and `target_points_clamped=false`.

## Round 3: Contracts/Docs (Agent-C)
Files:
- `/Users/collier/projects/apps/bittr-tess-vetter/docs/cli_contracts.md`
- `/Users/collier/projects/apps/bittr-tess-vetter/README.md`
- `/Users/collier/projects/apps/bittr-tess-vetter/docs/quickstart.rst` (if FPP examples exist)

Tasks:
- [ ] Document new fields/defaults and migration behavior.
- [ ] Add clear note: reduction operates on folded/windowed points.
- [ ] Add practical guidance: binning is most distinct at `target_points` ~100–500.

Acceptance:
- [ ] Docs align with CLI help and emitted payloads.

## Round 4: Test Coverage (Agent-C)
Files (minimum):
- `/Users/collier/projects/apps/bittr-tess-vetter/tests/validation/test_triceratops_fpp_replicates.py`
- `/Users/collier/projects/apps/bittr-tess-vetter/tests/cli/test_btv_fpp_cli_contract.py`

Tasks:
- [ ] Unit tests for `downsample`/`bin`/`none` behavior.
- [ ] Determinism test for binning.
- [ ] Bin floor test: `target_points > n_points_windowed` clamps gracefully.
- [ ] Sparse-window test: `1 <= n_points_windowed < 20` continues with warning and `low_window_point_count=true`.
- [ ] Bin-edge boundary test (interior half-open bins, final closed bin).
- [ ] Robust low-occupancy fallback test (`n_bin < 2`).
- [ ] CLI contract tests for new options + provenance fields.
- [ ] Regression: requested `mc_draws` honored regardless of timeout.
- [ ] Regression: output emits canonical `target_points`.
- [ ] Regression: legacy `max_points` inputs are accepted with deprecation warnings and canonicalized `target_points` output while preserving downsample behavior unless overridden.
- [ ] Regression: `--max-points` + `--target-points` precedence (equal warn, different fail).
- [ ] Regression: trace precedence shows canonical `target_points` source and legacy alias metadata when equal values are supplied.
- [ ] Regression: alias-only `max_points` input emits source `legacy_max_points_alias`.
- [ ] Regression: trace metadata schema keys are present and correctly populated.
- [ ] Regression: `point_reduction=none` uses ignored-input trace sources (`target_points_ignored_for_none` / `legacy_max_points_alias_ignored_for_none`) as applicable.
- [ ] Regression: `point_reduction=none` + explicit `target_points` warn-and-ignore behavior.
- [ ] Regression: `point_reduction=none` + explicit `max_points` canonicalizes then warn-and-ignore.
- [ ] Regression: `point_reduction=none` emits `effective_config.target_points=null` and `target_points_clamped=false`.
- [ ] Regression: empty-window input (`n_points_windowed == 0`) fails with `no_windowed_points` for all modes (`downsample|bin|none`).
- [ ] Regression: `bin_stat=median` + `bin_err=propagate` fails with actionable error.
- [ ] Regression: `flux_err_0` collapse method/provenance fields emitted as specified.
- [ ] Regression: synthetic fixture guarantees `bin` vs `downsample` `time_folded_used` divergence at `target_points=100`, with both outputs valid.
- [ ] Regression: bin time coordinate uses bin centers.
- [ ] Regression: runtime metric default/null semantics are enforced.

Acceptance:
- [ ] New tests fail pre-change and pass post-change.

## Round 5: Usability Runbook (Agent-D)
Tasks:
- [ ] Run end-to-end TOI-5739 matrix with fixed seeds:
  - `point_reduction`: `downsample`, `bin`
  - `target_points`: `100, 250, 500, 1000, 1500`
  - seeds: `13, 29, 41, 67, 101`
- [ ] Compare:
  - runtime (single dedicated host class; discard 1 warmup run per pair; 3 repeats; compare median)
  - replicate spread
  - disposition stability
  - provenance field completeness against spec schema

Deliverable:
- [ ] `/Users/collier/projects/apps/bittr-tess-vetter/working_docs/binning/TOI5739_binning_usability.md`

## Final Review Gate (Agent-D)
- [ ] Findings ordered by severity with file/line refs.
- [ ] Confirm no hidden runtime mutation remains.
- [ ] Confirm CLI/help/docs consistency.
- [ ] Confirm measurable gate outcomes:
  - `bin` median runtime <= 1.5x `downsample` at each target-point setting
  - disposition agreement (`same final class label`, `bin` vs `downsample`) >= 90% across 25 paired runs
  - CLI help examples/alias migration notes present and executable

## Release Gate
Commands:
- [ ] `uv run ruff check .`
- [ ] `uv run mypy src/tess_vetter`
- [ ] `uv run pytest`

Metadata:
- [ ] Update `CHANGELOG.md`
- [ ] Bump version in `pyproject.toml`, `src/tess_vetter/__init__.py`, `CITATION.cff`
- [ ] Tag and push release

## Handoff Artifacts
- [ ] `/Users/collier/projects/apps/bittr-tess-vetter/working_docs/binning/IMPLEMENTATION_SUMMARY.md`
- [ ] `/Users/collier/projects/apps/bittr-tess-vetter/working_docs/binning/REVIEW_REPORT.md`
- [ ] `/Users/collier/projects/apps/bittr-tess-vetter/working_docs/binning/TOI5739_binning_usability.md`
