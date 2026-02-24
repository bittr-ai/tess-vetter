# Changelog

This project follows semantic versioning (SemVer).

## Unreleased

- No unreleased changes.

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
