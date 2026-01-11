# Next Physics Audit Priorities

This note captures the **next most critical physics-audit targets** based on blast radius (how many downstream modules depend on the item) and risk of silent, high-impact physics errors (units, time base, sign conventions, masking).

If another agent claims one of these items, de-prioritize it and move to the next entry.

## Top priorities (current)

1) `src/bittr_tess_vetter/compute/primitives.py`
   - Shared low-level numerical helpers; a unit/sign mismatch here propagates widely.

2) `src/bittr_tess_vetter/compute/detrend.py`
   - Detrending is a primary “physics risk surface”: overfitting can erase or invert transit signals; window-size and mask semantics matter.

3) `src/bittr_tess_vetter/compute/bls_like_search.py`
   - Period/epoch/duration conventions, phase folding, and SNR proxies are easy to get subtly wrong and hard to notice via unit tests alone.

4) `src/bittr_tess_vetter/api/report.py` + `src/bittr_tess_vetter/api/evidence.py`
   - Evidence aggregation is where units (ppm vs fraction), signs, and “what is being summarized” can drift across layers.

5) `src/bittr_tess_vetter/api/exovetter.py`
   - Exovetter wrapper needs careful confirmation that time bases, durations, and mask windows match the intended checks.

6) `src/bittr_tess_vetter/activity/primitives.py` + `src/bittr_tess_vetter/activity/result.py`
   - Variability/rotation metrics are commonly misinterpreted if timescales, cadence gaps, and normalization are inconsistent.

7) `src/bittr_tess_vetter/io/cache.py`
   - Cache key semantics + stored metadata can create silent provenance/time-base mismatches if not explicit.

8) `src/bittr_tess_vetter/utils/tolerances.py` (+ `src/bittr_tess_vetter/api/tolerances.py`)
   - Central numeric tolerances directly affect alias/consistency logic; off-by-unit errors can silently widen/narrow acceptance bands.

9) `src/bittr_tess_vetter/api/catalog.py` + `src/bittr_tess_vetter/api/catalogs.py`
   - Wrapper semantics (skips vs failures vs warnings) affect completeness and downstream interpretation.

10) `src/bittr_tess_vetter/compute/model_competition.py` (+ `src/bittr_tess_vetter/api/model_competition.py`)
   - Model scoring/selection often mixes units and noise models; audit for consistent likelihood normalization and guardrails.
