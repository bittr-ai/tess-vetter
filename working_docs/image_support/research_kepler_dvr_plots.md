# Kepler DVR/DVS Plot Research

## Summary

This document catalogs the standard diagnostic plots in Kepler Data Validation Reports (DVRs) and Summaries (DVSs), identifies critical plots for vetting, and notes TESS-specific adaptations.

**Primary Sources:**
- [Twicken et al. 2018 - Kepler Data Validation I](https://iopscience.iop.org/article/10.1088/1538-3873/aab694)
- [NASA Exoplanet Archive DVS Documentation](https://exoplanetarchive.ipac.caltech.edu/docs/DVOnePageSummaryPageCompanion-dr25-V7.html)
- [TESS SPOC Pipeline Documentation](https://heasarc.gsfc.nasa.gov/docs/tess/data-handling.html)

---

## 1. Standard DVS One-Page Summary Plots (A-H)

| Label | Plot Type | Description | Vetting Purpose |
|-------|-----------|-------------|-----------------|
| **A** | Full Time-Series | Complete flux light curve across all quarters; triangles mark transits (red=rolling band artifact coincidence, blue=clean) | Detect inter-quarter systematics, instrumental artifacts |
| **B** | Phased Full-Orbit | Phase-folded light curve at TCE period; secondary eclipse positions marked | Assess transit model fit; detect out-of-transit variability |
| **C** | Secondary Eclipse | Strongest secondary eclipse candidate; shows depth (ppm), phase, MES | Identify eclipsing binary false positives |
| **D** | Phased Transit-Only | Zoomed primary transit with hourly resolution | Assess transit shape, symmetry, ingress/egress |
| **E** | Whitened Transit | Whitened/filtered transit with residuals; shows MES, SNR, chi-squared | Evaluate model fit quality after noise removal |
| **F** | Odd-Even Comparison | Separate odd and even transit depths | Detect eclipsing binary (different eclipse depths) |
| **G** | Centroid Offset | PRF-derived centroid offsets per quarter; 3-sigma circle | Identify background eclipsing binaries |
| **H** | DV Analysis Table | Fit parameters + diagnostic statistics | Quantitative vetting metrics |

---

## 2. Full DVR Additional Diagnostic Plots

### 2.1 Pixel-Level Diagnostics
| Plot Type | Description | Purpose |
|-----------|-------------|---------|
| **Difference Images** | Out-of-transit minus in-transit pixel data | Localize transit signal source |
| **PRF-Fit Centroids** | Pixel Response Function centroid fits | Sub-pixel transit source location |
| **Pixel Correlation Maps** | Transit signal correlation across aperture | Identify off-target signals |

### 2.2 Statistical Diagnostic Plots
| Plot Type | Description | Purpose |
|-----------|-------------|---------|
| **Bootstrap Distribution** | Null hypothesis MES distribution | False alarm probability assessment |
| **Ghost Diagnostic** | Core vs. halo aperture correlation | Detect optical ghost artifacts |
| **Rolling Band Severity** | RBA severity levels (0-4) vs. transit times | Identify instrumental false positives |

### 2.3 Multi-Planet System Plots
| Plot Type | Description | Purpose |
|-----------|-------------|---------|
| **Multiple Planet Search** | Light curve after signal removal | Detect additional planets |
| **Period Ratio Analysis** | Candidate period comparisons | Identify EB primary/secondary pairs |

---

## 3. Critical Plots for Vetting Decisions

### Tier 1: Essential (Must Have)
1. **Phased Transit (D/E)** - Core signal characterization
2. **Odd-Even Comparison (F)** - Primary EB discriminator
3. **Secondary Eclipse (C)** - EB detection via thermal/reflection
4. **Centroid Offset (G)** - Background contamination check

### Tier 2: Important
5. **Full Time-Series (A)** - Context and artifact identification
6. **Difference Images** - Pixel-level source verification
7. **Phase-Folded Full Orbit (B)** - Out-of-transit behavior

### Tier 3: Supporting
8. **Bootstrap Statistics** - Significance quantification
9. **Ghost Diagnostic** - Optical artifact check
10. **Rolling Band Diagnostic** - Kepler-specific artifact

---

## 4. TESS-Specific Adaptations

### Inherited from Kepler (Same Tests)
- Centroid offset analysis
- Odd/even transit comparison
- Secondary eclipse search
- Difference imaging
- Ghost diagnostic
- Bootstrap false alarm probability

### TESS Modifications
| Aspect | Kepler | TESS |
|--------|--------|------|
| **Observation Duration** | 4 years continuous | 27-day sectors (typically 1-13 sectors) |
| **Rolling Band** | Major issue (1-year artifacts) | Not applicable (different electronics) |
| **Pixel Scale** | 4 arcsec/pixel | 21 arcsec/pixel (more blending) |
| **Centroid Precision** | Higher (smaller pixels) | Lower (larger pixels, more crowding) |
| **Difference Images** | Per-quarter | Per-sector FFI-based |

### TESS-Specific Considerations
- **Larger pixels** = higher contamination probability = centroid tests more critical
- **Shorter baseline** = fewer transits = bootstrap statistics less robust
- **Full-frame images** = broader community tools (TESS-plots, LATTE)

---

## 5. Implementation Priority for TESS Vetter

### Priority 1: Core Diagnostic Plots
1. **Phase-folded transit** - Essential for any vetter
2. **Odd/even transit comparison** - Primary EB discriminator
3. **Secondary eclipse search** - EB and planet characterization
4. **Full light curve with transits marked** - Context

### Priority 2: Pixel-Level Diagnostics
5. **Centroid offset visualization** - Critical for TESS (large pixels)
6. **Difference images** - Source localization
7. **Pixel correlation / aperture check** - Contamination assessment

### Priority 3: Statistical Diagnostics
8. **Bootstrap / significance metrics** - Quantitative confidence
9. **Model fit residuals** - Transit shape quality
10. **Ghost diagnostic indicators** - Artifact flagging

### Priority 4: Multi-Signal Analysis
11. **Secondary planet search plots** - After signal removal
12. **Period ratio diagnostics** - Multi-planet validation

---

## 6. Key Diagnostic Metrics (Numeric)

From DVR analysis tables, vetters examine:

| Metric | What It Tests | Red Flag Values |
|--------|---------------|-----------------|
| `tce_max_mult_ev` | Detection significance (MES) | < 7.1 (weak) |
| `tce_robstat` | Robust detection statistic | Low values |
| `tce_depth_err` | Transit depth uncertainty | > 50% of depth |
| `tce_ror` | Rp/R* ratio | > 0.3 (too large) |
| `tce_dor` | a/R* ratio | < 1 (unphysical) |
| `tce_dicco_msky` | Centroid offset (arcsec) | > 3 sigma |
| `tce_dikco_msky` | KIC centroid offset | > 3 sigma |
| `tce_albedo_stat` | Secondary eclipse albedo | > 1 (unphysical) |
| `tce_ptemp_stat` | Planet temperature | > 4000K (likely EB) |
| `tce_fwm_stat` | Flux-weighted centroid motion | Significant shift |
| `tce_ghost_core_ha` | Ghost diagnostic | Strong correlation |

---

## 7. Community Vetting Tools Reference

| Tool | Focus | Plots Generated |
|------|-------|-----------------|
| **TESS-plots** | TESS FFI vetting | Difference images, phase diagrams, odd/even, secondaries |
| **LATTE** | TESS vetting | TPF movies, centroid, nearby sources, momentum dumps |
| **Lightkurve** | General | Phase-folded, periodograms, TPF visualization |
| **eleanor** | TESS FFI | Light curves, TPF cutouts |

---

## 8. References

1. Twicken et al. (2018) - "Kepler Data Validation I" - PASP 130, 064502
2. Li et al. (2018) - "Kepler Data Validation II" - Companion paper on centroid/difference imaging
3. NASA Exoplanet Archive DVS Documentation
4. TESS Science Data Products Description Document (EXP-TESS-ARC-ICD-TM-0014)
5. [TESS-plots GitHub](https://github.com/mkunimoto/TESS-plots)
