# Module Review: `catalogs/exofop_target_page.py` + `catalogs/exofop_toi_table.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is high leverage

ExoFOP is the community “state of follow-up” source for TESS candidates. These modules are lightweight, but they affect:
- “already researched” vs “novel” heuristics
- whether we should spend time on certain candidates
- provenance/caching behavior across sessions (offline-first operation)

The main risks are *not* unit conversions; they are **caching correctness** and “false negatives” from brittle HTML parsing.

## File: `catalogs/exofop_toi_table.py`

### Function: `fetch_exofop_toi_table`

- Inputs / outputs:
  - downloads the ExoFOP TOI table in pipe-delimited form (`output=pipe`)
  - returns an `ExoFOPToiTable` containing normalized headers and raw string rows
- Caching:
  - in-process cache `_CACHE` with TTL
  - disk cache at `<cache_root>/exofop/toi_table.pipe`
  - uses a temp file + rename for atomic writes
  - disk cache respects TTL via file `mtime`
- Parsing:
  - header normalization is conservative and stable (lowercase, punctuation stripping, whitespace collapsing, `%`→`pct`)
  - `entries_for_tic` searches common TIC header variants: `tic_id`, `tic`, `ticid`
- Known failure regimes:
  - a partially downloaded or malformed pipe table will parse to empty headers/rows; this will behave as “no entries” (false negative). Disk caching could then persist that, but only if it gets written.
  - code writes whatever it downloads; it does not validate that the parsed header includes a TIC column before caching.

### Tests

- Existing tests:
  - `tests/catalogs/test_exofop_disk_cache.py` verifies disk cache is used (network is not called) and TTL=0 disables disk cache.

## File: `catalogs/exofop_target_page.py`

### Function: `fetch_exofop_target_summary`

- Inputs / outputs:
  - fetches ExoFOP target page HTML (`target.php?id=<tic>`)
  - extracts badge counts from “grid headers” (e.g., Imaging Observations, Files)
  - emits `ExoFOPTargetSummary` with:
    - `grid_badges` (raw badge counts keyed by grid title)
    - `followup_counts` mapped to a small stable set (`imaging`, `spectroscopy`, `time_series`, `files`)
    - `flags` for parse anomalies
- Caching:
  - in-process dict cache `_CACHE` keyed by TIC, TTL-based
  - disk cache at `<cache_root>/exofop/target_pages/<tic>.json`
  - importantly: treats “no badges” as poisoned cache and will not use/write it (prevents false “no followup”)
- Parsing brittleness:
  - uses a specific regex over HTML; ExoFOP markup changes can break it.
  - the `exofop_parse_no_badges` flag is an important guardrail; callers should surface it rather than treating empty followup as meaningful.

### Tests

- Existing tests:
  - `tests/catalogs/test_exofop_disk_cache.py` verifies disk cache usage and TTL semantics.

## Fixes / follow-ups (non-blocking)

- Consider validating that the TOI table parse produced at least a minimal expected header set before persisting to disk, to reduce risk of caching a transient failure.
- Consider adding a unit test for `_parse_grid_badges` with a representative HTML snippet (to catch regex drift).

