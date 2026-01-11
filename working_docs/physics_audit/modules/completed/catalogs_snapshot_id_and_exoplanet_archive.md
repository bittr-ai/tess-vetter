# Module Review: `catalogs/snapshot_id.py` + `catalogs/exoplanet_archive.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is high leverage

- Snapshot IDs are the backbone of reproducibility for any cached catalog data; subtle format drift can break provenance linking.
- Exoplanet Archive queries provide “known planets” ephemerides used for recovery/vetting; the most critical physics hazard is time-base conversion (BJD ↔ BTJD) and depth units.

## File: `catalogs/snapshot_id.py`

### Purpose / semantics

Defines an immutable snapshot ID format:

`catalog:<name>:<version>:<asof_yyyymmdd>:<sha256prefix>`

This is a different snapshot-ID scheme than the simpler `<name>:<version>:<checksum_prefix>` used by `catalogs/store.py`. That’s OK as long as each system is used consistently (but it is a potential integration footgun).

### Function: `generate_snapshot_id`

- Units: not applicable (provenance).
- Correctness:
  - uses SHA-256 over the provided `data: bytes`, takes an 8-hex prefix.
  - validates all components through `SnapshotComponents`.
- Risks:
  - `SHA256_PREFIX_LENGTH=8` is intentionally short; collision risk is low but not zero (acceptable for “prefix routing”, less ideal for security).
  - `_validate_asof_date` uses a simplified month/day table (allows Feb 29 always; leap-year logic is not strict). This is a validity guardrail, not a calendrical library; acceptable.

### Function: `parse_snapshot_id` / `validate_snapshot_id`

- Correctness:
  - strict parsing: 5 parts, prefix must be `catalog`.
  - `validate_snapshot_id` returns `False` rather than raising.

### Tests

- Existing tests:
  - `tests/catalogs/test_snapshot_id.py` is extensive (name/version/date rules, prefix rules, determinism, invalid cases).

## File: `catalogs/exoplanet_archive.py`

### Core physics conventions

- Time base:
  - Converts `pl_tranmid` (documented as BJD at Exoplanet Archive) to **BTJD** via:
    - `BTJD = BJD - 2457000`
  - This matches TESS convention used elsewhere in the codebase (`t0_btjd`).
- Depth:
  - PS table `pl_trandep` is treated as **percent** and converted to ppm via `depth_ppm = depth_pct * 10000`.
  - This is consistent with “percent → fraction (÷100) → ppm (×1e6)” = ×1e4.

### Function: `_bjd_to_btjd`

- Correctness:
  - applies constant offset and passes through `None`.
- Risks:
  - Assumes input is on the same BJD system as BTJD convention; Exoplanet Archive values are typically BJD_TDB. This is standard for exoplanet tables; acceptable for typical TESS use, but should be documented as “BJD_TDB → BTJD”.

### Function: `_parse_ps_record`

- Output semantics:
  - Essential fields: requires `pl_name` and `pl_orbper`.
  - `t0` becomes 0.0 if missing/unavailable after conversion (this is a notable policy choice; downstream should treat `t0=0` as “missing”).
- Units:
  - `duration_hours` reads `pl_trandur` (hours).

### Function: `_parse_toi_record` (candidate ingestion)

- Notes:
  - `pl_trandurh` is documented as hours; consistent with `duration_hours`.
  - `pl_trandep` is used as ppm directly for TOI table (per Exoplanet Archive conventions).

### Function: `get_known_planets`

- Semantics:
  - Queries PS by `tic_id` (string field `"TIC {tic_id}"`) or by hostname prefix if `target` provided.
  - Optionally queries TOI table by numeric `tid` if a TIC is known.
  - Filters candidates: skips CONFIRMED and FALSE_POSITIVE; includes candidate-like statuses.
  - Sorts by period.
- Risks / footguns:
  - If PS query returns missing `pl_tranmid`, the returned `t0=0.0` is not a valid BTJD and can mislead downstream if not checked.
  - Mixed provenance: PS and TOI values may not be internally consistent (expected; best-available sources differ).

### Tests

- Existing tests:
  - No dedicated unit tests found for `catalogs/exoplanet_archive.py` (most catalog tests are offline/fixture-based for Gaia/SIMBAD/snapshot/store).
- Suggested tests:
  - Unit tests with mocked `requests.get` verifying:
    - BJD→BTJD conversion
    - percent→ppm conversion for PS depth
    - TOI duration field (`pl_trandurh`) is treated as hours
    - missing `pl_tranmid` yields `t0=0.0` (and ideally a warning flag)

## Fixes / follow-ups (non-blocking)

- Consider representing missing `t0` as `None` rather than `0.0` in `KnownPlanet` (would be a backwards-compat change).
- Add targeted unit tests for `catalogs/exoplanet_archive.py` to prevent time-base or depth-unit regressions.

