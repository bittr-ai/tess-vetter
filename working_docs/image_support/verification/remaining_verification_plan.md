# Remaining Plot Verification Plan

**Status:** COMPLETE (for implemented plots) - Verified 2026-01-20

---

## Summary

All implemented plots now have generators in `scripts/verify_plots.py` and have been visually verified.

| Plot | Function | Status |
|------|----------|--------|
| V01 | `plot_odd_even` | PASS |
| V02 | `plot_secondary_eclipse` | PASS |
| V03 | `plot_duration_consistency` | PASS |
| V04 | `plot_depth_stability` | PASS |
| V05 | `plot_v_shape` | PASS |
| V06 | `plot_nearby_ebs` | PASS |
| V07 | `plot_exofop_card` | PASS |
| V08 | `plot_centroid_shift` | PASS |
| V09 | `plot_difference_image` | PASS |
| V10 | `plot_aperture_curve` | PASS |
| V11 | `plot_modshift` | PASS |
| V12 | `plot_sweet` | PASS |
| V13 | `plot_data_gaps` | PASS |
| V15 | `plot_asymmetry` | PASS |
| V16 | `plot_model_comparison` | PASS |
| V17 | `plot_ephemeris_reliability` | PASS |
| V18 | `plot_sensitivity_sweep` | PASS |
| V19 | `plot_alias_diagnostics` | PASS |
| V20 | `plot_ghost_features` | PASS |
| V21 | `plot_sector_consistency` | PASS |
| - | `plot_phase_folded` | PASS |
| - | `plot_full_lightcurve` | PASS |
| - | `plot_transit_fit` | PASS |
| - | `plot_vetting_summary` (DVR) | PASS |

---

## Fixes Applied

1. **V21 annotation overlap** - Moved chi2 p-value from upper-right to upper-left to avoid legend overlap
2. **V09 marker overlap** - When target == max_depth pixel, now shows single marker with combined label

---

## Notes

V18 (`plot_sensitivity_sweep`) is now implemented and covered by `scripts/verify_plots.py` (`v18_sensitivity_sweep.png`).

---

## Running Verification

```bash
uv run --extra plotting python scripts/verify_plots.py
```

Output: 22 PNG files in `verification_plots/`

---

## Notes

- All plots follow standard astrophysics conventions
- origin="lower" correctly applied to all image plots
- Colorbars present on all 2D image plots
- Error bars shown where applicable
- Legends positioned to avoid overlap with data/annotations
