# Citations v2 — Recovery, TTV, Stitching (agent: time-series)

**Author**: codex (opus-style task split)  
**Focus**: recovery workflow helpers, TTV track search, and stitching utilities.

## Recovery helpers missing citations (P0)

The main recovery entrypoints (`recover_transit`, etc.) are cited, but these helpers currently report **0 refs**:

- `src/bittr_tess_vetter/api/recovery.py:125` `prepare_recovery_inputs`
- `src/bittr_tess_vetter/api/recovery.py:439` `stack_transits`

Reason to fix:
- These are workflow-building primitives; users may call them directly to build custom detrend/stack pipelines.
- Attaching the same recovery references improves traceability without changing algorithms.

## TTV track search lacks citations entirely (P0)

These are algorithmic and currently have 0 refs:

- `src/bittr_tess_vetter/api/ttv_track_search.py:85` `run_ttv_track_search`
- `src/bittr_tess_vetter/api/ttv_track_search.py:159` `run_ttv_track_search_for_candidate`
- plus helper heuristics (`identify_observing_windows`, `estimate_search_cost`, …)

Recommendation:
- Decide on the “canonical” TTV references you want to cite for this feature and attach them at least to the `run_*` entrypoints.
- If you don’t want to claim novelty, it’s still helpful to cite a foundational TTV reference and/or the Kepler DV TTV handling literature.

## Stitching: likely OK to remain uncited (P1)

- `src/bittr_tess_vetter/api/stitch.py:134` `stitch_lightcurves`
- `src/bittr_tess_vetter/api/stitch.py:235` `stitch_lightcurve_data`

These are more “data plumbing” than a specific published algorithm. If you want citations anyway, you’d be citing pipeline/normalization conventions rather than a named method.

