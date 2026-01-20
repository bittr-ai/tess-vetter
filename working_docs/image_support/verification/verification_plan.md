# Plot Verification Plan

## Quick Start
```bash
uv run python scripts/verify_plots.py
# Outputs to: verification_plots/
```

---

## Verification Checklist

### V01: Odd-Even Depth
**Reference:** Kepler DVR "Odd/Even Depth Comparison" panel
| Criterion | Expected |
|-----------|----------|
| Odd transits | Red points with error bars |
| Even transits | Green points with error bars |
| Mean lines | Horizontal dashed lines for each set |
| Y-axis | Depth (ppm), not flux |
| X-axis | Epoch number |

### V02: Secondary Eclipse
**Reference:** Kepler DVR "Secondary Eclipse" panel
| Criterion | Expected |
|-----------|----------|
| Primary window | Shaded region near phase 0 |
| Secondary window | Shaded region near phase 0.5 |
| Depth annotation | Text showing measured secondary depth |
| Phase range | 0 to 1 (or -0.5 to 0.5) |

### V04: Depth Stability
**Reference:** Transit depth vs time plots in discovery papers
| Criterion | Expected |
|-----------|----------|
| Per-epoch depths | Points with error bars |
| Mean depth | Horizontal dashed line |
| Expected scatter | Shaded band around mean |
| X-axis | Time (BTJD), not epoch |

### V05: V-Shape
**Reference:** Kepler DVR "Model Fit" panel
| Criterion | Expected |
|-----------|----------|
| Binned data | Points with error bars |
| Model overlay | Trapezoid or limb-darkened model |
| Transit visible | Clear dip at phase 0 |
| Flat bottom | Distinguishable from V-shape |

### V08: Centroid Shift
**Reference:** Kepler DVR "Centroid Offsets" panel
| Criterion | Expected |
|-----------|----------|
| Background image | TPF reference frame |
| Out-of-transit centroid | Distinct marker (cyan) |
| In-transit centroid | Distinct marker (magenta) |
| Target position | Cross marker (red) |
| Origin | Lower-left (origin="lower") |
| Colorbar | Flux units (e-/s) |

### V09: Difference Image
**Reference:** Kepler DVR "Difference Image" panel
| Criterion | Expected |
|-----------|----------|
| Pixel grid | 2D depth map |
| Target marker | Visible at target pixel |
| Max depth marker | Visible at brightest pixel |
| Colormap | Diverging (positive/negative) |
| Origin | Lower-left |

### V21: Sector Consistency
**Reference:** Multi-sector analysis in TESS papers
| Criterion | Expected |
|-----------|----------|
| Per-sector values | Bars or points with error bars |
| Weighted mean | Horizontal reference line |
| Outliers | Visually distinct (red) |
| X-axis | Sector numbers |

### Phase-Folded Transit
**Reference:** Standard in all transit papers
| Criterion | Expected |
|-----------|----------|
| Phase range | Centered on transit (typically -0.1 to 0.1) |
| Binned data | Larger points overlaid on scatter |
| Transit shape | Clear U-shape at phase 0 |
| Y-axis | Normalized flux (near 1.0) |

### DVR Summary
**Reference:** Kepler Data Validation Report format
| Criterion | Expected |
|-----------|----------|
| Multi-panel | 4-8 panels in grid layout |
| Light curve panel | Full time series |
| Phase-folded panel | Transit detail |
| Odd-even panel | Depth comparison |
| Metrics panel | Text summary of key values |

---

## Kepler DVR Reference Standards

Key conventions from Kepler/TESS DVRs:
1. **Flux direction**: Transits go DOWN (flux decreases)
2. **Phase convention**: Transit at phase 0, secondary at ~0.5
3. **Image origin**: Lower-left (astronomical convention)
4. **Error bars**: Always show uncertainties
5. **Units**: ppm for depths, normalized flux for light curves
6. **Colormaps**: Viridis/plasma for flux, diverging for difference images

---

## Automated Tests

The test suite covers:
- `tests/test_plotting/test_checks.py` - V01-V05 plots
- `tests/test_plotting/test_pixel.py` - V08-V10 plots
- `tests/test_plotting/test_extended.py` - V16-V21 plots
- `tests/test_plotting/test_report.py` - DVR summary
- `tests/test_plotting/test_integration.py` - API exports

Run: `uv run pytest tests/test_plotting/ -v`

---

## Visual Verification Process

1. Run `scripts/verify_plots.py`
2. Open each PNG in `verification_plots/`
3. Check against criteria above
4. For Kepler DVR comparison, see: https://exoplanetarchive.ipac.caltech.edu/docs/KSCI-19105-002-DV.pdf
