# Plot Verification Notes

**Date:** 2026-01-20  
**Verifier:** Codex CLI (GPT-5.2)  
**Plan reference:** `working_docs/image_support/verification/verification_plan.md`

---

## Commands Run

### Plot generation
- `uv run --extra plotting -- python scripts/verify_plots.py`

### Test suite
- `uv run --group dev -m pytest -q`

Notes:
- `pytest` is not available globally in this environment; running via `uv run --group dev` worked.
- `scripts/verify_plots.py` requires matplotlib; running without `--extra plotting` failed with `ModuleNotFoundError: No module named 'matplotlib'`.

---

## Outputs

**Directory:** `working_docs/image_support/verification/verification_plots/`

Generated files are enumerated in the table under “Manual Visual Inspection (All Generated PNGs)”.

---

## Visual Checklist Results (Observed vs Expected)

### V01: Odd-Even Depth (`verification_plots/v01_odd_even.png`)
- Depth (ppm) on y-axis: PASS
- Epoch on x-axis: PASS
- Error bars present: PASS
- Mean lines present (dashed): PASS
- **Odd/Even colors:** FAIL vs `verification_plan.md` (plot shows Odd=red, Even=green)

### V02: Secondary Eclipse (`verification_plots/v02_secondary_eclipse.png`)
- Primary window shaded near phase ~0: PASS
- Secondary window shaded near phase ~0.5: PASS (window spans ~0.4–0.6)
- Depth annotation present: PASS (“Secondary: 50 ppm”)
- Phase range 0–1: PASS

### V04: Depth Stability (`verification_plots/v04_depth_stability.png`)
- Per-epoch depths with error bars: PASS
- Mean depth line (dashed): PASS
- Expected scatter band (shaded): PASS
- X-axis uses time (BTJD): PASS (matplotlib displays an offset “+2.459e6”)

### V05: V-Shape (`verification_plots/v05_v_shape.png`)
- Binned data points with error bars: PASS
- Model overlay (trapezoid): PASS
- Transit dip centered at phase 0: PASS
- tF/tT annotation present: PASS (`tF/tT = 0.60`)

### V08: Centroid Shift (`verification_plots/v08_centroid_shift.png`)
- Background image present: PASS
- Out-of-transit centroid marker present (cyan): PASS
- In-transit centroid marker present (magenta): PASS
- Target marker present (red “+”): PASS
- Origin appears lower-left: PASS (axes increase upward/right; labeled Row/Column)
- Colorbar present and labeled in flux units: PASS (“Flux (e-/s)”)

### V09: Difference Image (`verification_plots/v09_difference_image.png`)
- 2D pixel depth map visible: PASS
- Diverging colormap: PASS
- Origin appears lower-left: PASS
- Target marker visible: PASS
- **Max depth marker visible:** PASS, but overlaps the target marker in this synthetic example (both at same pixel)

### V21: Sector Consistency (`verification_plots/v21_sector_consistency.png`)
- Per-sector values shown with error bars: PASS
- Weighted mean line shown: PASS
- Outliers visually distinct: PASS (none flagged in this example; legend includes “Outlier”)
- X-axis is sector numbers: PASS
- **Minor layout issue:** the `chi^2 p-val: 0.450` annotation is partially obscured by the legend.

### Phase-folded transit (`verification_plots/phase_folded.png`)
- Phase range centered on 0: PASS (≈ -0.15 to 0.15)
- Raw scatter + binned overlay: PASS
- Transit dip visible: PASS
- Y-axis normalized flux near 1: PASS

### DVR summary (`verification_plots/dvr_summary.png`)
- Multi-panel layout present: PASS (currently 4 panels)
- Light curve panel present: PASS
- Phase-folded panel present: PASS
- Odd-even panel present: PASS
- Metrics summary present: PASS
- Note: This is not the full 8-panel DVR layout described in the full spec; it appears to be a minimal summary variant.

---

## Automated Test Results

`uv run --group dev -m pytest -q`: PASS

Warnings observed (non-blocking):
- `lightkurve` optional submodule warning (oktopus)
- `pkg_resources` deprecation warnings (pytransit)
- a few runtime warnings from test fixtures (NaN median; exovetter “No cadences found…”)

---

## Follow-ups / Recommended Fixes

1. **Unify V01 odd/even colors across docs + code**
   - `verification_plan.md` expects Odd=red, Even=green.
   - `plotting_spec.md` previously standardized Odd=green, Even=red.
   - Pick one convention and update the other document (or make verify plan match current implementation).

2. **Improve V21 annotation placement**
   - Move `chi^2 p-val` text away from the legend (e.g., place text at upper-left and legend at upper-right or lower-right).

3. **Consider handling marker overlap in V09**
   - When `target_pixel == max_depth_pixel`, draw a single combined marker or offset one marker slightly for legibility.

---

## Addendum (post-fixes)

**Date:** 2026-01-20  
**Changes applied after initial run:**
- V21 chi-squared annotation moved to upper-left (no longer overlaps legend).
- V09 target/max marker overlap resolved by drawing a single combined marker when `target_pixel == max_depth_pixel`.
- `scripts/verify_plots.py` now also generates `transit_fit.png` (covers `plot_transit_fit`).

**Commands re-run:**
- `uv run --extra plotting -- python scripts/verify_plots.py`
- `uv run --group dev --extra plotting -m pytest -q tests/test_plotting`

**Updated outputs now include (in `verification_plots/`):**
- `transit_fit.png`
- `full_lightcurve.png`
- `v03_duration_consistency.png`
- plus the original set from the initial run

**Resolved discrepancy (V18):**
- Implemented and exported `plot_sensitivity_sweep` (V18) and added `v18_sensitivity_sweep.png` generation in `scripts/verify_plots.py`.

---

## Full Validation Run

### Lint / typecheck / tests
- `uv run --group dev -m pytest -q`: PASS (warnings only from external deps/fixtures)
- `uv run --group dev -m ruff check src tests`: PASS
- `uv run --group dev -m mypy src/bittr_tess_vetter/plotting`: PASS

### Optional dependency behavior
- Importing `bittr_tess_vetter.api` without matplotlib: PASS
- Attempting to access plotting symbols without matplotlib raises `ImportError`: PASS
- Importing `bittr_tess_vetter.api` with `--extra plotting` and accessing plotting callables: PASS

### End-to-end smoke (real pipeline)
- Ran `api.vet_candidate(lc, candidate)` on a synthetic light curve and rendered:
  - `verification_plots/dvr_summary_from_vet_candidate.png`

---

## Manual Visual Inspection (All Generated PNGs)

**Run:** `uv run --extra plotting -- python scripts/verify_plots.py`  
**Directory:** `working_docs/image_support/verification/verification_plots/`  
**Result:** All images reviewed manually; no blocking issues found.

| File | Status | Notes |
|------|--------|-------|
| `v01_odd_even.png` | PASS | Odd=red, Even=green; mean lines + error bars present. |
| `v02_secondary_eclipse.png` | PASS | Primary/secondary windows shaded; secondary depth annotation present. |
| `v03_duration_consistency.png` | PASS | Observed/expected bars + expected error bar + ratio annotation. |
| `v04_depth_stability.png` | PASS | Mean line + expected scatter band + per-epoch error bars; BTJD axis OK. |
| `v05_v_shape.png` | PASS | Trapezoid overlay + binned error bars + `tF/tT` annotation. |
| `v06_nearby_ebs.png` | PASS | Target + nearby EBs + search-radius circle; sep labels readable. |
| `v07_exofop_card.png` | PASS (minor) | Notes text can overflow the card box for long strings; consider wrapping/clipping. |
| `v08_centroid_shift.png` | PASS | origin="lower"; markers + colorbar label correct; legend readable. |
| `v09_difference_image.png` | PASS | Diverging cmap + colorbar; combined marker works when target==max-depth. |
| `v10_aperture_curve.png` | PASS | Depth vs radius curve + error bars; baseline at 0 shown. |
| `v11_modshift.png` | PASS | Primary/secondary markers and annotations present; legend OK. |
| `v12_sweet.png` | PASS | Data + sinusoid overlay; legend indicates SNR. |
| `v13_data_gaps.png` | PASS | Coverage bars + threshold + max-missing annotation; legend OK. |
| `v15_asymmetry.png` | PASS | Left/right shaded bins + mean lines + asymmetry annotation. |
| `v16_model_comparison.png` | PASS | Data + three model overlays + winner annotation. |
| `v17_ephemeris_reliability.png` | PASS | Null curve + on-ephemeris marker + p-value annotation. |
| `v18_sensitivity_sweep.png` | PASS (minor) | Long variant labels make y-axis text wide; OK but consider tightening label formatting. |

### Post-polish updates

- `plot_exofop_card` now wraps long notes onto multiple lines (with clipping) to avoid overflowing the card box.
- `plot_sensitivity_sweep` now abbreviates variant labels (e.g., `sigma_clip_4` → `sc4`, `running_median_0.5d` → `rm0.5d`) to keep the y-axis compact.
| `v19_alias_diagnostics.png` | PASS | Best harmonic highlighted + legend present. |
| `v20_ghost_features.png` | PASS | Difference image + aperture contour + colorbar + in/out depth annotation. |
| `v21_sector_consistency.png` | PASS | p-value annotation no longer overlaps legend; weighted mean shown. |
| `phase_folded.png` | PASS | Raw scatter + binned overlay; transit centered at phase 0. |
| `full_lightcurve.png` | PASS | Transit spans visible; legend OK. |
| `transit_fit.png` | PASS | Data + model overlay + fit parameter box; axis labels OK. |
| `dvr_summary.png` | PASS | Summary layout renders cleanly for the selected panels (A/B/D/H). |
