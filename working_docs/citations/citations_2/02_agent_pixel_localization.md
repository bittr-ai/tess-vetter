# Citations v2 — Pixel Localization & PRF (agent: pixel)

**Author**: codex (opus-style task split)  
**Focus**: pixel-localization user entrypoints + PRF-based hypothesis scoring.

## Status: PRF citations are now mostly in good shape

The old report called out PRF/pixel scoring as missing citations. In current code:

- `src/bittr_tess_vetter/api/pixel_prf.py` wraps PRF construction + scoring functions with citations (e.g. Bryson PRF lineage).
- `src/bittr_tess_vetter/api/wcs_localization.py` has `@cites(...)` and a `REFERENCES` list covering difference imaging / centroid methods and WCS foundations.

## Remaining gap: orchestration functions are citation-empty (P0)

These are the user-facing localization orchestrators that currently report **0 refs** via `get_function_references`:

- `src/bittr_tess_vetter/api/pixel_localize.py:145` `localize_transit_host_single_sector`
- `src/bittr_tess_vetter/api/pixel_localize.py:360` `localize_transit_host_single_sector_with_baseline_check`
- `src/bittr_tess_vetter/api/pixel_localize.py:467` `localize_transit_host_multi_sector`

Why this is important:
- These are the “headline” pixel-host identification tools.
- Researchers will cite *these*, not the internal `pixel.wcs_localization` helper.

### Suggested minimal citation set for these orchestrators

You already cite these sources elsewhere; the missing piece is attaching them to these orchestrators:

- Difference imaging / centroid offsets (Kepler DV lineage): Twicken et al. 2018 (already in `api/wcs_localization.py`)
- Difference-image localization diagnostics: Bryson et al. 2013 (already in `api/wcs_localization.py`)
- PRF model construction / fitting: Bryson et al. 2010 (already used in `api/pixel_prf.py`)
- FITS WCS: Greisen & Calabretta 2002; Calabretta & Greisen 2002 (already in `api/wcs_localization.py`)

## Secondary issue: duplicate-doc warnings (FYI, not citation-related)

When building docs, Sphinx emits “duplicate object description” warnings for some Pydantic fields (e.g. `LocalizationResult.*`). This is documentation tooling noise; it doesn’t affect citation correctness, but it may obscure citation issues in logs.

