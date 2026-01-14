# Citations v2 — API Surface Inventory (agent: api-scan)

**Author**: codex (opus-style task split)  
**Focus**: `bittr_tess_vetter.api.*` callables that currently return *no* references via `get_function_references`.

## Machine scan summary

Source: `working_docs/citations/citations_2/api_submodules_reference_scan.json`

- Total callables defined in `bittr_tess_vetter.api.*`: **72**
- Callables with `n_refs == 0`: **39**

Full triage table: `working_docs/citations/citations_2/missing_citations_table.md`

## High-signal findings (things worth fixing)

### A) Orchestration functions missing citations (P0)

These are “top of funnel” functions for users that aggregate multiple cited checks/primitives, but have no citations attached themselves:

- `src/bittr_tess_vetter/api/vet.py:147` `vet_many`
- `src/bittr_tess_vetter/api/lc_only.py:356` `vet_lc_only`
- `src/bittr_tess_vetter/api/pixel.py:317` `vet_pixel`
- `src/bittr_tess_vetter/api/catalog.py:232` `vet_catalog`
- `src/bittr_tess_vetter/api/exovetter.py:230` `vet_exovetter`
- `src/bittr_tess_vetter/api/pixel_localize.py:145` `localize_transit_host_single_sector`
- `src/bittr_tess_vetter/api/pixel_localize.py:360` `localize_transit_host_single_sector_with_baseline_check`
- `src/bittr_tess_vetter/api/pixel_localize.py:467` `localize_transit_host_multi_sector`
- `src/bittr_tess_vetter/api/ttv_track_search.py:85` `run_ttv_track_search`
- `src/bittr_tess_vetter/api/ttv_track_search.py:159` `run_ttv_track_search_for_candidate`
- `src/bittr_tess_vetter/api/recovery.py:125` `prepare_recovery_inputs`
- `src/bittr_tess_vetter/api/recovery.py:439` `stack_transits`

Why it matters:
- Researchers often want to cite *the orchestrator*, not chase internal calls.
- Your `get_function_references` feature becomes much more valuable if orchestrators expose their bibliographies.

Suggested patterns (choose one):
- Decorate orchestrators directly with `@cites(...)`, reusing the module’s `REFERENCES` list references.
- Or wrap them like `vet_lc_only = cites(...)(vet_lc_only)` to match the style used in other wrappers.

### B) Possibly-OK but worth deciding (P1)

- `src/bittr_tess_vetter/api/stitch.py:134` `stitch_lightcurves`
- `src/bittr_tess_vetter/api/stitch.py:235` `stitch_lightcurve_data`

These are algorithmic enough that a citation could be defensible (stitching/normalization choices), but it’s also reasonable to leave uncited and treat as “engineering glue”.

## Low-signal / should remain uncited

These show up as “missing” because the scan is mechanical, but citations are not expected:

- Citation system utilities: `bittr_tess_vetter.api.references.*` (e.g. `cite`, `cites`, `generate_bibtex`, …)
- Evidence/hash/serialization helpers: `api/canonical.py`, `api/evidence*.py`
- Introspection helpers: `api/primitives_catalog.py`, `api/pipeline.py` (`list_checks`, `describe_checks`)
- Cache helper: `api/triceratops_cache.py:get_disposition`
- Mask utility: `api/transit_masks.py:get_out_of_transit_mask_windowed`

## Recommendation

Treat citations as part of the “public contract”:

- **P0**: Add citations to the orchestration functions listed above.
- **P1**: Decide explicitly whether stitching utilities should remain citationless.
- **P2**: Optionally annotate utility functions with “No citations required” in docstrings to avoid future confusion.

