# Next Physics Audit Priorities (Post-Agent-5)

This note captures the **next most critical physics-audit targets** based on blast radius (how many downstream modules depend on the item) and risk of silent, high-impact physics errors (units, time base, sign conventions, masking).

If another agent claims one of these items, de-prioritize it and move to the next entry.

## Top priorities

1) `src/bittr_tess_vetter/pixel/cadence_mask.py`
   - Central cadence hygiene for pixel-level algorithms; any masking convention mistake propagates through most pixel checks.

2) `src/bittr_tess_vetter/api/mlx.py` (+ its dependency chain)
   - The facade claims MLX is optional, but it currently imports MLX-linked modules eagerly; this can create environment-dependent failures and silently force CPU fallback.

3) `src/bittr_tess_vetter/catalogs/spatial.py` + `src/bittr_tess_vetter/catalogs/crossmatch.py`
   - Core separation/crossmatch logic drives crowding and host-ambiguity priors; unit/geometry issues here have large downstream impact.

4) `src/bittr_tess_vetter/catalogs/gaia_client.py` + `src/bittr_tess_vetter/catalogs/simbad_client.py` + `src/bittr_tess_vetter/catalogs/store.py`
   - Catalog provenance + caching + snapshot semantics; subtle coordinate/magnitude/provenance mismatches can bias host hypotheses and vetting context.

5) `src/bittr_tess_vetter/domain/target.py`
   - Defines stellar parameters and derived quantities (e.g., density) used in multiple guardrails and consistency checks.

6) `src/bittr_tess_vetter/validation/ephemeris_specificity.py` (+ `src/bittr_tess_vetter/api/ephemeris_specificity.py`)
   - Tolerance/alias logic is easy to get wrong (time bases, units, epoch conventions); errors here can mis-associate candidates.

7) `src/bittr_tess_vetter/validation/prefilter.py` (+ `src/bittr_tess_vetter/api/prefilter.py`)
   - Prefilters often gate expensive analysis; physics/units mistakes become systematic selection bias.

8) `src/bittr_tess_vetter/compute/transit.py` + `src/bittr_tess_vetter/compute/primitives.py`
   - Core compute surfaces are common sources of silent unit mismatches (hours vs days, ppm vs fraction).

9) `src/bittr_tess_vetter/api/report.py` + `src/bittr_tess_vetter/pixel/report.py`
   - Evidence aggregation/reporting is where sign conventions and summary-stat semantics can drift from underlying checks.

10) `src/bittr_tess_vetter/transit/timing.py` + `src/bittr_tess_vetter/transit/result.py` + `src/bittr_tess_vetter/recovery/pipeline.py`
   - Timing/recovery is vulnerable to time-base, cadence, and stacking/epoch-reference mistakes that are hard to detect post-hoc.

