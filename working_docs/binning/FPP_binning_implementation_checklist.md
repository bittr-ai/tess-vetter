# FPP Binning Implementation Checklist

## Scope
Implement `point_reduction=bin` as a first-class alternative to current downsampling for FPP preprocessing, with full CLI/config/provenance/test coverage.

## Branch
- Target branch: `main` (or `feature/fpp-binning` if desired)

## Team Topology
- `Agent-A` (Core runtime): `validation/triceratops_fpp.py`
- `Agent-B` (CLI/policy/provenance): `cli/fpp_cli.py` + command help
- `Agent-C` (Tests/contracts/docs): `tests/*`, docs
- `Agent-D` (Reviewer): full code review + usability check

## Round 0: Lock Requirements
- [ ] Confirm field names: `point_reduction`, `target_points`, `bin_stat`, `bin_err`.
- [ ] Confirm defaults: `downsample`, `target_points=1500`.
- [ ] Confirm no timeout-driven mutation of `mc_draws` or point-reduction knobs.
- [ ] Confirm migration rule: accept `max_points` input alias, emit `target_points` in output.

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
- [ ] Preserve scalar `flux_err_0` behavior explicitly (representative value from reduced points).
- [ ] Ensure `n_points_raw/windowed/used` are accurate.

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
- [ ] Thread policy into runtime call.
- [ ] Update `--help` with concise examples.

Acceptance:
- [ ] `resolution_trace` includes reduction fields.
- [ ] Invalid combinations fail with actionable errors.

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
- [ ] Unit tests for `bin`/`downsample`/`none` behavior.
- [ ] Determinism test for binning.
- [ ] Bin floor test: `target_points > n_points_windowed` clamps gracefully.
- [ ] CLI contract tests for new options + provenance fields.
- [ ] Regression: requested `mc_draws` honored regardless of timeout.
- [ ] Regression: output emits canonical `target_points`.

Acceptance:
- [ ] New tests fail pre-change and pass post-change.

## Round 5: Usability Runbook (Agent-D)
Tasks:
- [ ] Run end-to-end TOI-5739 matrix with fixed seeds:
  - `point_reduction`: `downsample`, `bin`
  - `target_points`: `100, 250, 500, 1000, 1500`
- [ ] Compare:
  - runtime
  - replicate spread
  - disposition stability
  - output clarity for agents

Deliverable:
- [ ] `/Users/collier/projects/apps/bittr-tess-vetter/working_docs/binning/TOI5739_binning_usability.md`

## Final Review Gate (Agent-D)
- [ ] Findings ordered by severity with file/line refs.
- [ ] Confirm no hidden runtime mutation remains.
- [ ] Confirm CLI/help/docs consistency.

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
