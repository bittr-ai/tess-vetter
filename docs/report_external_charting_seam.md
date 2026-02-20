# External Charting Seam (bittr-plot-agnostic)

This document defines the stable JSON seam consumed by downstream adapter/charting repos (for example `bittr-plot-agnostic`).

## Scope

- Producer: `ReportData.to_json()` in `tess_vetter.report`.
- Consumer: external adapters that transform report payloads into chart-specific models.
- Non-goal: HTML renderer behavior in this repo.

## Guaranteed Payload Envelope

Top-level payload keys are contract keys:

- `schema_version` (string)
- `summary` (object)
- `plot_data` (object)
- `custom_views` (object)
- `payload_meta` (object)

Current producer value for `schema_version` is `2.0.0`.

Guaranteed semantics:

- Payload is JSON-safe (no `NaN`/`Inf` values).
- Producer serializes with `exclude_none=True`; `None` is emitted as missing key.
- `summary` and `plot_data` are additive domains: consumers must tolerate missing optional subkeys.
- `payload_meta` is deterministic metadata, including hashes and compatibility hints.

## Custom Views and JSON Pointer Rules

Custom views reference data via RFC 6901 JSON Pointer paths.

- Pointer source fields: `custom_views.views[].chart.series[].x|y|y_err.path`
- Allowed roots only: `/summary/...` and `/plot_data/...`
- Disallowed root examples: `/payload_meta/...`, `/schema_version`, any non-absolute path.

Adapter expectations:

- Resolve pointers exactly per RFC 6901 unescaping (`~1` -> `/`, `~0` -> `~`).
- If any path is invalid or unresolved, producer degrades that view:
- `quality.status = "unavailable"`
- `quality.flags` includes `INVALID_PATH` and/or `UNRESOLVED_PATH`
- Treat degraded view as non-renderable, but continue processing other views.

## Determinism Guarantees

Producer guarantees deterministic hash computation for cache/identity use:

- `payload_meta.summary_hash`: SHA-256 of canonicalized `summary`
- `payload_meta.plot_data_hash`: SHA-256 of canonicalized `plot_data`
- `payload_meta.custom_views_hash`: SHA-256 of canonicalized custom views
- `payload_meta.custom_view_hashes_by_id`: per-view deterministic hashes

Canonicalization details:

- JSON serialized with sorted keys and fixed separators.
- Non-finite floats normalized to `null` before hashing.
- Custom views normalized by sorting `views` by `id` before full-block hashing.

## Compatibility and Versioning Expectations

For external adapter repos:

- Gate hard compatibility on `schema_version` major version.
- Treat unknown additional keys as forward-compatible (ignore unless needed).
- Do not use content hashes (`*_hash`) as schema compatibility signals; they are cache/content identity only.
- Use `payload_meta.summary_version`, `plot_data_version`, and `custom_views_version` for block-specific parser evolution.
- Use `payload_meta.contract_version` and required/missing metric maps when adapter behavior depends on strict check-metric completeness.

Recommended adapter policy:

- Fail fast only when top-level envelope is invalid or `schema_version` major is unsupported.
- Soft-degrade on missing optional fields and unavailable custom views.
