# Citations v2 — Catalog/FPP/Exovetter (agent: catalogs)

**Author**: codex (opus-style task split)  
**Focus**: catalog-based vetting and optional-extras APIs.

## Catalog orchestration is citation-empty (P0)

- `src/bittr_tess_vetter/api/catalog.py:232` `vet_catalog` reports 0 refs.

The same file has:
- `REFERENCES = [...]` with at least PRSA/ExoFOP/TOI context refs
- other functions already decorated with `@cites(...)`

Recommendation:
- Attach the module’s references to `vet_catalog`, since it’s the entrypoint researchers will call.

## Exovetter orchestration is citation-empty (P0)

- `src/bittr_tess_vetter/api/exovetter.py:230` `vet_exovetter` reports 0 refs.

The module already cites key vetting literature (Thompson/Coughlin) on the specific wrapped vetters.
For user-facing clarity, `vet_exovetter` should likely carry the same citation set.

## FPP / TRICERATOPS

Status:
- `src/bittr_tess_vetter/api/fpp.py` maintains a `REFERENCES` list and uses `@cites(...)` for its public entrypoints.
- Cache helper `src/bittr_tess_vetter/api/triceratops_cache.py:get_disposition` has 0 refs (reasonable: it’s a cache helper, not an algorithm).

## Takeaway

The remaining work here is not “find missing PRSA/TRICERATOPS papers”; it’s **attach existing citations to the top-level orchestrator functions** so the runtime citation introspection matches user workflows.

