# Module Review: `api/references.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this matters

This module is the central citation registry. It does not implement astrophysics calculations, but it directly impacts:
- provenance completeness (tools exposing citation metadata),
- stable IDs for references across the codebase,
- host-side auditing/reporting that relies on `__references__`.

## File: `api/references.py`

- Defines:
  - `Reference` (immutable bibliographic entry with `to_dict()` and `to_bibtex()`)
  - `Citation` (`Reference` + optional context string)
  - `cite(...)` helper
  - `cites(...)` decorator that attaches `__references__` to callables
  - helper functions for collecting and exporting citations (module/function introspection)
- Registry:
  - Maintains an internal `_REGISTRY` of `Reference` objects keyed by `id` via the `reference(...)` decorator.
- Safety/constraints:
  - Canonical JSON compliance is enforced indirectly by rejecting NaN/Inf in canonicalization utilities (not in this module).
  - This module is introspection-heavy but does not do any I/O or network.

## Physics/unit considerations

None. This module handles citation metadata only.

## Fixes / follow-ups

No physics correctness issues identified in this module.

