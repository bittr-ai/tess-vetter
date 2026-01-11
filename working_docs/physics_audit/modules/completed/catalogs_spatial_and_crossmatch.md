# Module Review: `catalogs/spatial.py` + `catalogs/crossmatch.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is high leverage

These modules sit on the “astro-context” boundary:
- `SpatialIndex` is used to support cone searches and neighbor finding; if separation math is wrong, *crowding/host ambiguity* becomes wrong.
- `crossmatch` determines novelty/known-object matching and a first-pass contamination estimate; mistakes here can bias follow-up decisions.

## File: `catalogs/spatial.py`

### Function: `SpatialIndex.cone_search`

- Location: `src/bittr_tess_vetter/catalogs/spatial.py`
- Public API? yes (imported from `bittr_tess_vetter.catalogs`)
- Units + conventions:
  - inputs `ra, dec` in degrees; `radius_arcsec` in arcseconds
  - internally converts to unit-sphere Cartesian coordinates
- Physics correctness:
  - Uses a chord-distance query in Cartesian space (`chord = 2 sin(angle/2)`), which is a correct way to do spherical cone search with a Euclidean k-d tree.
  - Handles RA wrap-around via normalization to `[0, 360)` before conversion.
- Numerical stability:
  - Uses the stable small-angle angular distance formula in `_angular_distance_rad`.
  - Clamps `chord/2` to `[0,1]` to prevent domain errors in `asin` due to floating error.
- Known failure regimes:
  - Does not validate that catalog coordinate RA is already normalized; negative/large RA values in the catalog will still be converted consistently, but upstream should ideally keep RA in a standard range.
  - Requires cadence axis 0? not applicable; this is purely geometry.

### Function: `SpatialIndex.angular_separation`

- Output: arcseconds
- Method: converts to unit vectors then uses the chord→angle mapping; correct and stable for small/large separations.

### Tests

- Existing tests:
  - `tests/catalogs/test_spatial.py` has comprehensive coverage:
    - construction, determinism, RA wrap-around, pole behavior, known separations, and cone search correctness.

## File: `catalogs/crossmatch.py`

### Function: `angular_separation_arcsec`

- Location: `src/bittr_tess_vetter/catalogs/crossmatch.py`
- Units:
  - inputs degrees; output arcseconds
- Method:
  - haversine formula over spherical coordinates; correct and stable for small separations.
  - The formula is periodic in ΔRA (through `sin(ΔRA/2)`), so RA wrap-around behaves correctly without explicit normalization.

### Function: `compute_dilution_factor`

- Units/semantics:
  - magnitude-based flux ratios (logarithmic magnitudes) with target flux normalized to 1.0
  - `neighbor_flux_total = Σ 10^((m_target - m_neighbor)/2.5)`
  - dilution factor returned is `F_target / (F_target + ΣF_neighbors)`
- Physics assumptions:
  - This is a *photometric* dilution proxy that assumes all neighbors contribute equally (no PSF/aperture weighting).
  - Useful as a coarse prior/flag; not a substitute for pixel-level aperture-family checks.

### Function: `assess_contamination`

- Behavior:
  - scans all catalog entries and treats any non-known-object entry as a “neighbor star”
  - computes separations and includes neighbors with `sep <= search_radius_arcsec` and `sep > 0.1` (the latter avoids self-matches)
  - returns nearest-neighbor separation, Δmag vs target (if available), and a dilution estimate from all neighbor magnitudes
- Risks/notes:
  - The `sep > 0.1 arcsec` self-match exclusion is heuristic; if a real companion is within 0.1″ it will be excluded. For TESS-scale use this is likely acceptable, but it is a real edge case for high-resolution catalogs.
  - The contamination estimate uses only the entries provided by snapshot(s); completeness depends on which snapshot IDs are passed.

### Function: `find_known_object_matches` + `determine_novelty_status`

- Behavior:
  - matches entries with `object_type` in `{TOI, CONFIRMED, FP, EB}` within radius
  - novelty is `"known"` if any match is `CONFIRMED` or `TOI`, otherwise `"ambiguous"` if only FP/EB, else `"novel"`.
- Assumptions:
  - this is a policy choice, not a physics formula.

### Tests

- Existing tests:
  - `tests/catalogs/test_crossmatch.py` covers:
    - angular separation sanity (including polar behavior)
    - dilution factor math
    - contamination neighbor selection and ordering
    - known-object matching, novelty status, and snapshot-id validation.

## Fixes / follow-ups (none blocking)

- Consider documenting (or parameterizing) the `0.1 arcsec` “self-match exclusion” threshold in contamination assessment.
- Consider consolidating separation logic to use `catalogs.spatial.SpatialIndex.angular_separation` for a single canonical implementation (not required for correctness; both are fine).

