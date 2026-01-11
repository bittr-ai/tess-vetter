# Module Review: `utils/canonical.py` + `utils/caps.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this matters

These utilities underpin reproducibility and response stability:
- canonical JSON/hashes are used for deterministic IDs and cache keys,
- response caps prevent pathological payload sizes that can break host tools.

They are not astrophysics calculations, but they directly affect whether physics outputs are reliably attributable and transportable.

## File: `utils/canonical.py`

- `canonical_json(obj)`:
  - Deterministic JSON bytes: sorted dict keys, UTF-8, no whitespace.
  - Floats are rounded to `FLOAT_DECIMAL_PLACES` (10), NaN/Inf raise `ValueError`.
  - Numpy arrays/scalars are converted to Python lists/scalars.
  - Datetime/date objects serialize to ISO8601 strings.
- `canonical_hash(obj)` / `canonical_hash_prefix(obj, length=...)`:
  - SHA-256 digest of `canonical_json(obj)` (full or prefix).

## File: `utils/caps.py`

- Provides `_cap_list` and specific caps:
  - `cap_top_k`, `cap_variant_summaries`, `cap_neighbors`, `cap_plots`
- Semantics:
  - Truncates lists to max size; logs a warning on truncation.
  - No element re-ordering (always keeps first N).

## Fixes / follow-ups

No physics correctness issues identified in these utilities.

