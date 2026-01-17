# Spec: Harden Real-Candidate Validation Tutorial (TOI-5807.01 / TIC 188646744)

Owner: bittr-tess-vetter  
Status: Draft  
Target: `docs/tutorials/04-real-candidate-validation.ipynb` (+ optional small helper snippets)

## 1. Background

`docs/tutorials/04-real-candidate-validation.ipynb` is an end-to-end statistical validation tutorial (LC + pixel vetting + AO contrast curve + TRICERATOPS FPP/NFPP) for TOI-5807.01 (TIC 188646744).

Recent work added opt-in extended vetting diagnostics (V16–V21). During a dry-run of the extended preset on the tutorial’s bundled data:

- `V16` returned `ok` with a warning flag consistent with “non-transit model preferred” (`MODEL_PREFERS_NON_TRANSIT`).
- A methodological risk was identified in the notebook’s multi-sector pixel section: it downloads per-sector TPFs but runs vetting against a stitched multi-sector light curve, which can create timebase/mask mismatches if in-/out-of-transit masks are derived from the LC time series and then applied to the TPF time series.

This spec hardens the tutorial so the methodology is defensible and outputs are interpretable as *metrics-first* evidence (not new policy thresholds).

## 2. Goals

1. Make the multi-sector pixel localization section timebase-consistent (per-sector LC ↔ per-sector TPF).
2. Add explicit per-sector photometric consistency reporting (metrics-only; no new hard thresholds).
3. Add an “extended metrics” appendix section to the notebook that runs `preset="extended"` and reports V16–V21 outputs with careful interpretation guidance.
4. Keep the notebook runnable offline using the existing tutorial data directory; network-only steps stay optional and clearly labeled.

## 3. Non-goals

- Changing the default meaning of “full vetting” in the library (default preset remains 15 checks).
- Introducing new “PASS/FAIL” policy thresholds beyond what checks already report.
- Implementing new astrophysics algorithms inside `bittr-tess-vetter` (this is tutorial methodology + reporting).
- Re-validating the target in the literature sense (RV confirmation, dynamical confirmation, etc.).

## 4. Problems / Gaps to Address

### 4.1 Multi-sector pixel analysis likely mixes mismatched time series

In the current “Multi-Sector Pixel Localization” block, the code:

- downloads a sector-specific TPF (`tpf.time.btjd`), but
- calls `vet_candidate(lc, candidate, tpf=sector_tpf, ...)` using `lc` that is the stitched multi-sector LC spanning multiple sectors.

If the pixel checks build their in-transit / out-of-transit cadence masks from `lc.time`, those masks may not align to the TPF timebase. This can silently bias centroid/difference-image computations.

### 4.2 Per-sector normalization/detrending is not explicit

The notebook stitches PDCSAP across sectors. For bright stars, offsets and sector-to-sector trends can subtly bias depth/shape. A robust validation writeup typically:

- normalizes/detrends per sector, or
- explicitly demonstrates per-sector depth and shape stability.

### 4.3 Extended metrics exist but are not exercised in the validation narrative

The notebook’s conclusion asserts statistical validation based on baseline checks + FPP/NFPP. That may still be correct, but:

- extended metrics (V16–V21) can surface “robustness/systematics” indicators;
- for this TIC, `V16` flagged `MODEL_PREFERS_NON_TRANSIT`, which should be explicitly explained and reconciled (or bounded) to avoid overclaiming.

## 5. Proposed Changes

### 5.1 Fix multi-sector pixel analysis to use per-sector light curves

**Change:** For each sector in `SECTORS`, load the matching per-sector PDCSAP CSV (already present in tutorial data directory), construct a per-sector `LightCurve`, and pass that to `vet_candidate` alongside the sector TPF.

**Fallback behavior:** If per-sector LC file missing for a sector (or cannot be parsed), log a warning and either:

- skip that sector, or
- fall back to stitched LC with a prominent warning that timebase mismatch may occur.

**Output:** preserve the existing per-sector summary table, but ensure metrics are computed on time-consistent LC/TPF inputs.

### 5.2 Add per-sector depth consistency reporting (metrics-only)

**Change:** Add a new section after Step 6 (or within “Deep Dive”) that computes and prints a per-sector table:

- `sector`
- `n_points`, `time_span_days`, `n_transits_covered` (or coverage fraction)
- `depth_hat_ppm` and `depth_sigma_ppm` (or equivalent depth estimate)
- optionally: local baseline scatter / MAD ppm

**Implementation options (pick one):**

1. Use existing internal/LC vetting primitives (e.g., depth-stability/epoch-metrics style outputs) to produce a depth estimate per sector.
2. If no suitable primitive exists in the public API, implement a lightweight, transparent estimator in the notebook itself (e.g., median in-transit minus median out-of-transit within local windows per expected transit).

**Constraint:** Do not add new “PASS/FAIL” logic. The table is for interpretability and diagnostics.

### 5.3 Add an “Extended Metrics Appendix” to the notebook

**Change:** Add a section that runs:

- `vet_candidate(..., preset="extended")` (with the same inputs used for the baseline run)
- prints which of V16–V21 are `ok` vs `skipped` and dumps key metric keys.

**Specific guidance to include in text:**

- `V16 MODEL_PREFERS_NON_TRANSIT` is not a planet invalidation by itself; it’s a prompt to examine detrending/systematics and robustness.
- `V18` may skip without optional deps; make the dependency explicit in the notebook.
- `V21` may skip without host-provided per-sector measurements; explain that it’s reserved for integration scenarios where the host (pipeline) provides per-sector metrics.

### 5.4 Add a “Validation claim hygiene” paragraph

**Change:** Adjust language in “Validation Summary” to be precise:

- “Statistically validated per TRICERATOPS criteria (FPP<1%, NFPP<0.1%)” is appropriate **if**:
  - FPP/NFPP are computed with the AO contrast curve,
  - pixel localization is on-target across sectors,
  - there is no evidence of systematics dominating the signal (supported by new per-sector metrics + extended diagnostics discussion).

**Avoid:** implying “confirmed” or “dynamical confirmation”.

## 6. Acceptance Criteria

### Multi-sector pixel section

- For each analyzed sector, `vet_candidate` is called with a sector-specific `LightCurve` built from that sector’s CSV and a sector-specific `TPFStamp`.
- The notebook includes a clear note explaining why this matters (timebase/mask consistency).

### Per-sector consistency table

- Notebook prints a per-sector table with depth-related metrics and basic coverage stats.
- No new hard thresholds are introduced in the notebook (metrics-only presentation).

### Extended metrics appendix

- Notebook runs `preset="extended"` and presents V16–V21 status + key metrics.
- Notebook explicitly documents skip reasons for V18 (optional dep) and V21 (requires host-provided sector metrics).
- Notebook includes a short “how to interpret V16 flag” note with suggested follow-ups (e.g., per-sector detrend, robustness checks).

## 7. Follow-up Actions (Post-spec)

1. Implement the notebook edits.
2. Run the notebook (or a scripted “smoke run”) using the bundled data directory to confirm:
   - baseline vetting still runs,
   - per-sector tables generate without network,
   - extended preset runs (V16/V17/V19/V20 should run; V18/V21 may skip depending on environment).
3. Update any “expected output” blocks that change as a result of the methodology fix.

