# Module Review: `pixel/report.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is high leverage

Pixel-level vetting outputs are consumed by host apps and downstream evidence systems. Even though this module is “just reporting”, errors here can silently:
- drop/rename flags (changing interpretation),
- serialize non-JSON-safe structures (breaking host ingestion),
- lose provenance references (plots/manifests).

## File: `pixel/report.py`

### Helper: `_as_dict(obj)`

- Purpose: normalize various result payload types into a `dict[str, Any]`.
- Supported inputs:
  - `None` → `{}` (caller generally uses `None` upstream and bypasses `_as_dict`)
  - objects with `__dict__` (includes most dataclasses / pydantic models) → `dict(obj.__dict__)`
  - `dict` → `dict(obj)` copy
- JSON hygiene:
  - If a result contains `warnings` or `flags` stored as tuples, converts them to lists (JSON-friendly).
- Failure mode:
  - Raises `TypeError` for unsupported payload types (good: avoids silent serialization drift).

### Model: `PixelVetReport`

- Semantics: bundle of pixel diagnostics (no pass/warn/reject policy).
- Shape:
  - `centroid`, `difference_image`, `aperture_dependence`: optional `dict` payloads.
  - `quality_flags`: flattened list of warning/flag strings plus missing-result sentinels.
  - `manifest_ref`, `plot_refs`, `plots`: host-facing provenance slots.
- Pydantic config: `frozen=True`, `extra="forbid"` (prevents accidental schema widening).

### Function: `generate_pixel_vet_report(...)`

- Missing-result policy:
  - Adds `MISSING_*_RESULT` sentinels when a component is absent.
  - Appends stringified `centroid_result.warnings` and `aperture_result.flags` when present.
- Unit conventions:
  - No unit conversions; upstream pixel modules define the physics and units.

## Fixes / follow-ups

No physics correctness issues identified in this module.

