# Citations v2 — Initial Review (explore)

**Date**: 2026-01-14  
**Author**: codex (explore pass)  
**Goal**: Re-evaluate missing citations after v0.1.0 refactors (the original `working_docs/citations/MISSING_CITATIONS.md` is now stale).

## High-level state (what’s working)

- The project has a real citation system: `bittr_tess_vetter.api.references` provides `Reference`, `Citation`, `@cites(...)`, and runtime introspection (`get_function_references`, `collect_module_citations`).
- Most algorithmic public APIs already carry citations in one of two ways:
  - direct decorator usage (`@cites(...)`)
  - “wrapped re-export” usage (`foo = cites(...)(foo)`)
- Several API submodules maintain `REFERENCES = [...]` lists for module-level bibliography display.

## What changed vs the original report

The original report singled out PRF/pixel/systematics/reporting as missing. Those have mostly been addressed:

- `api/pixel_prf.py` now applies citations to PRF construction + hypothesis scoring functions.
- Pixel reporting and classic pixel primitives are now exposed via `api/…` re-exports that are citation-wrapped.
- Systematics proxy is exposed via `api/systematics.py` with citations (re-export wrapper style).

## What’s *actually* missing now (core insight)

The remaining “missing citations” are primarily **orchestration functions** that:
- aggregate other cited primitives/checks, but
- are not themselves decorated / citation-wrapped,
- so citation introspection reports **0 refs** for those top-level orchestrators.

This matters because researchers often start from orchestrators (`vet_*`, `localize_*`, `run_*`) and want “what literature supports this pipeline?”.

## Evidence artifacts produced in this pass

These machine outputs are in this folder and are meant to be used by follow-on agents:

- `working_docs/citations/citations_2/public_api_reference_scan.json`
  - Scans `bittr_tess_vetter.api.__all__` callables and checks citation coverage.
  - Result: only orchestration/introspection callables are citation-empty.
- `working_docs/citations/citations_2/api_submodules_reference_scan.json`
  - Scans *all* callables defined in `bittr_tess_vetter.api.*` submodules.
  - Result: 39 callables with zero refs (most are utilities; several are real orchestrators).
- `working_docs/citations/citations_2/missing_citations_table.md`
  - Triage table of the 39 “zero-ref” callables.

## Immediate “P0” candidates to cite (or explicitly declare “no-cite-needed”)

These are user-facing orchestration functions with **0 refs today**:

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

Note: many other “zero-ref” callables are intentionally citationless utilities (hashing/serialization/introspection), and should probably remain that way.

