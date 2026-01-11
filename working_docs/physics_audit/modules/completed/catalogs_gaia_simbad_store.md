# Module Review: `catalogs/gaia_client.py` + `catalogs/simbad_client.py` + `catalogs/store.py`

Applies: `working_docs/physics_audit/REVIEW_TEMPLATE.md`

## Why this is high leverage

These modules define the “astro-context priors” layer:
- Gaia cone search neighbors + RUWE flags drive multiplicity/crowding hypotheses.
- SIMBAD object typing and spectral parsing influence “early type / giant / binary” suitability checks.
- Snapshot store guarantees provenance and reproducibility for offline catalog usage.

Silent unit mistakes here can propagate into incorrect host-ambiguity flags and wrong priors for dilution/false-positive reasoning.

## File: `catalogs/gaia_client.py`

### Function: `_compute_separation_arcsec`

- Location: `src/bittr_tess_vetter/catalogs/gaia_client.py`
- Units:
  - inputs degrees (ICRS), output arcseconds
- Method:
  - haversine formula; appropriate for small angles and robust globally
  - periodic ΔRA handling is implicit via `sin(ΔRA/2)`
- Notes:
  - There is another separation helper in `catalogs/crossmatch.py` and `catalogs/spatial.py`; all are consistent in units/intent.

### Function: `GaiaSourceRecord.distance_pc`

- Units:
  - parallax in mas → distance pc = `1000 / parallax`
- Assumptions:
  - naive inversion (no Bayesian prior); appropriate as a convenience estimate, not for precision distance inference.

### Function: `GaiaClient._query_cone_search`

- Inputs:
  - `radius_arcsec` is converted to degrees for ADQL `CIRCLE`
- Output semantics:
  - constructs `GaiaNeighbor` entries with:
    - `separation_arcsec` computed via `_compute_separation_arcsec`
    - `delta_mag` computed as `g_mag - primary_mag` (positive means *fainter* than primary)
  - sorts neighbors by `(separation_arcsec, phot_g_mean_mag)`; if `phot_g_mean_mag` missing, uses sentinel `999.0`
- Physics correctness:
  - sorting rule matches docstring and makes sense for crowding triage.
  - RUWE threshold uses `> 1.4` (not >=), consistent with common usage.
- Known failure regimes / footguns:
  - `_resolve_tic_to_gaia()` is a stub returning `None`; `query_by_tic()` will always return an “empty result” with provenance and a warning log unless replaced upstream.
  - cone search ADQL orders results by brightness (`ORDER BY phot_g_mean_mag ASC`), but the returned list is resorted by separation anyway (good).

### Tests

- Existing tests:
  - `tests/catalogs/test_gaia_client.py` covers parsing, separation, RUWE flags, neighbor sorting, and provenance with offline fixtures.

## File: `catalogs/simbad_client.py`

### Function: `parse_spectral_type`

- Inputs:
  - raw spectral type string (e.g., `G0V`, `A5III`, `K2/3V`)
- Output semantics:
  - returns:
    - `luminosity_class` from regex (Ia/Ib/II/III/IV/V when present)
    - `is_early_type` if starts with O/B/A
    - `is_giant` if luminosity class indicates III/II/I (explicitly avoids IV/V)
- Risks:
  - spectral typing is a heuristic based on strings; SIMBAD entries can be messy and may include nonstandard formatting.
  - This is acceptable for coarse guardrails.

### Function: `classify_object_type`

- Semantics:
  - maps SIMBAD type codes into `is_star`, `is_binary`, `is_variable` flags using sets + some prefix heuristics.
- Risks:
  - the `is_star` heuristic treats many `*`-suffixed codes as stars; that’s reasonable but can over-include some exotic types.

### Tests

- Existing tests:
  - `tests/catalogs/test_simbad_client.py` exercises spectral parsing and object type classification with fixtures and unit cases.

## File: `catalogs/store.py`

### Class: `CatalogSnapshotStore`

- Purpose:
  - reproducible, versioned, checksummed catalog snapshots stored on disk
- Snapshot ID semantics:
  - format: `<name>:<version>:<checksum_prefix>`
  - checksum is SHA-256 over the stored `data.json` bytes
- Input validation:
  - name/version are restricted to filesystem-safe characters.
- Integrity:
  - `verify_checksum()` recomputes SHA-256 from stored bytes and compares to recorded checksum.
  - `exists()` checks whether metadata checksum starts with the snapshot prefix.

### Physics correctness

Not physics per se; it’s provenance. Correctness matters because it ensures catalog results are reproducible and auditable.

### Tests

- Existing tests:
  - `tests/catalogs/test_store.py` covers install/load/checksum/listing/errors using local temp storage and a local HTTP server.

## Fixes / changes (follow-ups)

No blocking physics issues found.

Follow-up candidates:
- Implement `_resolve_tic_to_gaia()` or clearly route TIC→Gaia resolution through an upstream crossmatch layer (to avoid “always empty” surprises when users call `query_by_tic`).
- Consider consolidating angular separation helpers across catalogs for a single canonical implementation (optional).

