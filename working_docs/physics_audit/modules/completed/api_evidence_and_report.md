# Module Review: `api/evidence.py` + `api/report.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is high leverage

These modules are “boundary surfaces” between the vetter’s internal result types and host-side storage/transport:
- `api/evidence.py` defines how `CheckResult` objects become JSON-like “evidence items”.
- `api/report.py` defines the public pixel-report surface (used by hosts to bundle pixel diagnostics).

If these layers drop fields, change units (ppm vs fraction), or serialize inconsistently, downstream interpretation breaks silently.

## File: `api/evidence.py`

### Helper: `_jsonable(value)`

- Goal: convert nested structures into JSON-serializable types.
- Supported conversions:
  - `None` → `None`
  - primitives (`str/int/float/bool`) → unchanged
  - `Enum` → `value.value`
  - `np.generic` → `.item()`
  - `np.ndarray` → `.tolist()`
  - `dataclass` → `asdict()` then recursively jsonable
  - `Mapping` → `{str(k): _jsonable(v)}`
  - `Sequence` (non-string/bytes) → list of jsonables
  - fallback → `str(value)` (best-effort stability)

### Function: `checks_to_evidence_items(checks)`

- Semantics:
  - Emits a list of dicts with keys:
    - `id`, `name`, `passed`, `confidence`
    - `metrics_only`: `True` if `details["_metrics_only"]` truthy OR if `passed is None`
    - `details`: jsonable copy of the details dict
- Physics/unit considerations:
  - No unit conversions occur here; this function must preserve the numeric semantics produced by the underlying checks (e.g., ppm vs fraction).
  - The only “policy-ish” behavior is determining `metrics_only`; it does not coerce pass/fail.

## File: `api/report.py`

- Role:
  - Public API wrapper that re-exports `PixelVetReport` and `generate_pixel_vet_report` from `pixel/report.py`.
  - Attaches citations to `generate_pixel_vet_report` via decorators (adds `__references__` metadata; no behavior change).
- Physics/unit considerations:
  - No unit conversions; report content is defined by pixel modules.

## Cross-references

- Pixel report implementation audit: `working_docs/physics_audit/modules/completed/pixel_report.md`

## Fixes / follow-ups

No physics correctness issues identified in these modules.

