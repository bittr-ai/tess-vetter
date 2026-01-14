# Citations v2 — Consolidated Missing Citations Inventory

**Date**: 2026-01-14  
**Author**: codex (consolidation)  
**Inputs**:
- `working_docs/citations/citations_2/00_explore_initial_review.md`
- `working_docs/citations/citations_2/01_agent_api_surface_inventory.md`
- `working_docs/citations/citations_2/02_agent_pixel_localization.md`
- `working_docs/citations/citations_2/03_agent_catalog_fpp_extras.md`
- `working_docs/citations/citations_2/04_agent_recovery_ttv_stitch.md`
- machine scans:
  - `working_docs/citations/citations_2/api_submodules_reference_scan.json`
  - `working_docs/citations/citations_2/missing_citations_table.md`

## Executive summary

The “missing citations” situation is **much better than the original report**. Most major algorithms in the public API already carry citations.

The current gap is concentrated in **orchestration entrypoints** that aggregate cited primitives/checks but have no citations attached themselves. This reduces the value of your citation introspection for real researcher workflows.

## P0: Fix these orchestration entrypoints (high leverage)

These callables currently report **0 refs** via `get_function_references`, but are likely to be directly cited/used by researchers:

### Vetting orchestrators
- `src/bittr_tess_vetter/api/vet.py:147` `vet_many`
- `src/bittr_tess_vetter/api/lc_only.py:356` `vet_lc_only`
- `src/bittr_tess_vetter/api/pixel.py:317` `vet_pixel`
- `src/bittr_tess_vetter/api/catalog.py:232` `vet_catalog`
- `src/bittr_tess_vetter/api/exovetter.py:230` `vet_exovetter`

### Pixel localization orchestrators
- `src/bittr_tess_vetter/api/pixel_localize.py:145` `localize_transit_host_single_sector`
- `src/bittr_tess_vetter/api/pixel_localize.py:360` `localize_transit_host_single_sector_with_baseline_check`
- `src/bittr_tess_vetter/api/pixel_localize.py:467` `localize_transit_host_multi_sector`

### Recovery helpers exposed as workflow primitives
- `src/bittr_tess_vetter/api/recovery.py:125` `prepare_recovery_inputs`
- `src/bittr_tess_vetter/api/recovery.py:439` `stack_transits`

### TTV track search entrypoints
- `src/bittr_tess_vetter/api/ttv_track_search.py:85` `run_ttv_track_search`
- `src/bittr_tess_vetter/api/ttv_track_search.py:159` `run_ttv_track_search_for_candidate`

## P1: Decide explicitly (OK either way)

Stitching utilities are uncited; you can leave them uncited without scientific awkwardness:

- `src/bittr_tess_vetter/api/stitch.py:134` `stitch_lightcurves`
- `src/bittr_tess_vetter/api/stitch.py:235` `stitch_lightcurve_data`

## P2: Ignore (utilities should remain uncited)

These show up as “missing citations” in mechanical scans but do not need literature references:

- `bittr_tess_vetter.api.references.*` (citation system itself)
- canonical hashing / evidence serialization utilities (`api/canonical.py`, `api/evidence*.py`)
- introspection (`api/primitives_catalog.py`, `api/pipeline.py`)
- cache helper (`api/triceratops_cache.py`)
- masking helper (`api/transit_masks.py`)

## Recommended implementation pattern (when you go to fix P0)

Goal: make `get_function_references(vet_pixel)` return the expected bibliography.

Minimal approach:
- Decorate the orchestrators with `@cites(...)` reusing the same references already listed in their module’s `REFERENCES` or already used in their underlying primitives.

Alternative (consistent with existing wrappers):
- Wrap via assignment: `vet_pixel = cites(REF1, REF2)(vet_pixel)`

## Appendix: machine-generated triage table

See: `working_docs/citations/citations_2/missing_citations_table.md`

