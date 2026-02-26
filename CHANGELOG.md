# Changelog

This project follows semantic versioning (SemVer).

## Unreleased

- No unreleased changes.

## 0.3.15 (2026-02-26)

- Simplified FPP v3 contract: `btv fpp plan` now stages inputs/artifacts only;
  runtime policy is supplied by `btv fpp run` / `btv fpp sweep`.
- Clarified FPP CLI help text across `plan`, `run`, `sweep`, `summary`, and
  `explain` with explicit option semantics for agent workflows.
- Fixed sweep scenario provenance so per-scenario outputs preserve the matrix
  scenario identifier (`provenance.runtime.scenario_id`).
- Added auto-scaled timeout behavior for FPP runs when `--timeout-seconds` is
  omitted, based on effective runtime policy complexity and replicate count.
- Enforced timeout as a single total budget across fallback attempts instead of
  resetting a full timeout per attempt.
- Bumped plan schema to `cli.fpp.plan.v2` to reflect the staging-only plan
  contract and avoid silent policy interpretation drift.
- Expanded CLI contract tests for timeout provenance, sweep scenario-id
  propagation, and total-budget timeout retry behavior.

## 0.3.14 (2026-02-26)

- Replaced legacy FPP command flow with a v3 command group:
  `btv fpp plan`, `btv fpp run`, `btv fpp sweep`, `btv fpp summary`,
  and `btv fpp explain`.
- Removed `btv fpp-prepare` and `btv fpp-run` in favor of explicit v3
  subcommands with clear error messaging.
- Added richer run provenance and first-class replicate reporting so run-level
  seed/config/outcome details are always available in output payloads.
- Updated default FPP behavior toward decision-grade execution by defaulting to
  network-enabled + stellar-auto resolution and enforcing stellar requirements
  for balanced/strict modes.
- Updated pipeline composition command token mapping to use `fpp plan`/`fpp run`
  and aligned CLI contract/pipeline tests with the new workflow.
- Refreshed CLI documentation for the v3 FPP contract and command semantics.

## 0.3.11 (2026-02-24)

- Fixed NumPy 2.x scalar-coercion regressions in vendored TRICERATOPS+
  limb-darkening lookup paths (`marginal_likelihoods.py`) and aperture flux
  integration (`triceratops.py`) so `btv fpp`/`btv fpp-run` avoid
  `only 0-dimensional arrays can be converted to Python scalars` failures.
- Expanded regression coverage to guard against reintroducing masked-array
  scalar assignment patterns in vendored TRICERATOPS+ code paths.
- Improved FPP CLI usability by supporting prepared-manifest mode directly on
  `btv fpp` and defaulting prepared-manifest runs to strict staged-artifact
  enforcement (`require-prepared`).

## 0.3.10 (2026-02-24)

- Fixed vendored TRICERATOPS Gaussian integrand behavior for SciPy `dblquad`
  under newer NumPy stacks by returning a scalar `float` when scalar inputs are
  provided.
- Added regression coverage for scalar-input behavior in
  `Gauss2D` used by `calc_depths` integration paths.

## 0.3.9 (2026-02-24)

- Fixed NumPy 2.x compatibility for vendored TRICERATOPS+ likelihood imports by
  injecting a `np.trapz` alias to `np.trapezoid` before `pytransit`
  module-level imports execute.
- This unblocks `triceratops` extra workflows that previously failed during
  import due to `pytransit` expecting `numpy.trapz` on newer NumPy stacks.

## 0.3.8 (2026-02-24)

- Added `--mast-timeout-seconds` for `btv measure-sectors`, `btv vet`,
  `btv report`, `btv fetch`, and `btv cache-sectors`.
- Added explicit timeout precedence for MAST calls:
  CLI flag > `BTV_MAST_TIMEOUT_SECONDS` > 60s default.
- Fixed `btv report --no-network` behavior so cache-only runs avoid MAST
  metadata calls and fail fast only when required cached data is missing.

## 0.3.7 (2026-02-24)

- Fixed `btv vet --split-plot-data` so sidecar extraction pulls from
  `results[*].raw.plot_data` (with top-level fallback) instead of producing
  empty sidecars in common runs.
- Added split provenance diagnostics on vet outputs:
  `plot_data_split_count` and `plot_data_split_schema_version`.
- Added CLI contract tests covering nested `raw.plot_data` split behavior and
  explicit `--no-split-plot-data` file-output behavior.

## 0.3.6 (2026-02-23)

- Fixed report sidecar payload projection to preserve schema-backed plot fields
  (not only `full_lc`), including `phase_folded`, `odd_even_phase`, and
  `secondary_scan`.
- Fixed `report_json` projection to preserve `payload_meta` and normalized
  verdict projection compatibility across top-level and summary fields.
- Fixed report vet-artifact coercion to accept top-level
  `known_planet_match` in valid `cli.vet.v2` payloads.
- Fixed code-mode execute error normalization to preserve non-preflight
  `result` payloads on failed responses.

## 0.3.5 (2026-02-23)

- Added CLI entrypoint ergonomics so all of these now work equivalently:
  `btv ...`, `tess-vetter ...`, and `python -m tess_vetter ...`.
- Added `btv doctor --profile vet` for environment/dependency preflight checks.
- Added early `btv vet` runtime dependency preflight for `lightkurve` with
  explicit remediation guidance.

## 0.3.4 (2026-02-23)

- Added a PyPI version badge to the top README badge row.
- Updated documentation and metadata URLs to the canonical repository: `https://github.com/bittr-ai/tess-vetter`.
- Adjusted CI so docs-only changes skip full CI, and Codecov upload runs even when the coverage gate fails.

## 0.3.1 (2026-02-23)

- Fixed API contract and preset override behavior in core execution paths.
- Hardened typing and runtime policy semantics across `code_mode` integration boundaries.
- Improved CI reliability and signal by narrowing required checks and moving heavier suites to scheduled runs.
- Stabilized flaky report timeout assertion behavior under variable CI scheduler latency.

## 0.3.0 (2026-02-23)

- Public release baseline for `tess-vetter`.
- Unified upstream API contract typing across core vetting and reporting surfaces.
- Introduced/expanded `code_mode` operation catalog, runtime policy enforcement, and MCP adapter coverage.
- Added extensive contract and regression test coverage for API/code_mode boundaries.
