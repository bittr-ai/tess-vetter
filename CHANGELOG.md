# Changelog

This project follows semantic versioning (SemVer).

## Unreleased

- No unreleased changes.

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
