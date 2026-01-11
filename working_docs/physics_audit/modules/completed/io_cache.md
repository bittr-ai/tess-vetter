# Module Review: `io/cache.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is high leverage

Caching is a provenance and correctness surface:
- cache keys determine whether results are inadvertently mixed across targets/sectors/products,
- cached objects must preserve time-base conventions (BTJD days) and metadata used downstream (tic_id/sector/cadence),
- eviction/clearing semantics affect reproducibility and debugging.

## File: `io/cache.py`

### Default cache location: `_default_cache_dir()`

Preference order:
1) `BITTR_TESS_VETTER_CACHE_DIR` (explicit path)
2) `BITTR_TESS_VETTER_CACHE_ROOT` + `/persistent_cache` (repo-local root)
3) `./.bittr-tess-vetter/cache/persistent_cache` under CWD

No physics logic; this only affects where cached data lives.

### Class: `SessionCache`

- Role:
  - Process-global, in-memory LRU cache for `LightCurveData` keyed by `data_ref`, with session namespacing.
  - Separate in-memory “computed products” cache keyed by `product_ref`.
- Namespacing:
  - Keys are stored as `"{session_id}:{data_ref}"` and `"{session_id}:{product_ref}"`.
- Eviction:
  - LRU eviction for both data and computed caches based on `OrderedDict` access/update order.
- Physics/provenance considerations:
  - Cache stores `LightCurveData` objects as-is; it does not modify arrays or units.
  - Correctness relies on callers using stable, non-colliding `data_ref` values (recommended: `api.lightcurve.make_data_ref`).

### Class: `PersistentCache`

- Role:
  - On-disk cache that persists across process restarts.
  - Stores values as `pickle` blobs and writes JSON metadata sidecars.
- Keying:
  - Uses SHA-256 hash of the string key for filenames; metadata JSON stores the original key for listing via `keys()`.
  - Callers should use deterministic key schemes (e.g., `lc:<tic_id>:<sector>:<flux_type>`).
- Metadata:
  - `created_at`, `accessed_at` timestamps, plus best-effort `tic_id` and `sector` extracted from cached objects via `getattr`.
  - Access time updated on `get()` (LRU-ish eviction basis).
- Eviction:
  - When at capacity, evicts least-recently accessed entries using `accessed_at` from metadata.
- Physics/provenance considerations:
  - Cache does not change the content of cached objects, but it can cause **mix-ups** if callers reuse keys across different time bases or products.
  - Because it uses pickle, cached values must be treated as local/trusted (host-controlled); this is not a physics issue but relevant to deployment posture.

## Fixes / follow-ups

No physics correctness issues identified in this module.

