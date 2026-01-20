# Vetting Check Plot Recommendations

Analysis of bittr-tess-vetter checks (V01-V21) with plot recommendations for each.

---

## Tier 1: Light Curve Checks (V01-V05)

### V01 - Odd/Even Depth
**Purpose:** Detect EBs masquerading as planets at 2x the true period.

| Plot | Data Source | Priority |
|------|-------------|----------|
| Phase-folded LC with odd/even transits colored differently | `epoch_depths_odd_ppm`, `epoch_depths_even_ppm` | **HIGH** |
| Bar chart: odd vs even depth with error bars | `depth_odd_ppm`, `depth_even_ppm`, `depth_err_odd_ppm`, `depth_err_even_ppm` | HIGH |
| Per-epoch depth scatter plot | `epoch_depths_odd_ppm`, `epoch_depths_even_ppm` vs epoch index | Medium |

**Scientific Value:** Directly shows the depth difference that would indicate an EB at 2x period.

---

### V02 - Secondary Eclipse
**Purpose:** Detect secondary eclipse indicating hot planet or EB.

| Plot | Data Source | Priority |
|------|-------------|----------|
| Phase-folded LC zoomed on phase 0.35-0.65 (secondary window) | LC flux, `search_window` bounds | **HIGH** |
| Full phase-folded LC with primary and secondary windows shaded | LC flux, highlight phase 0.5 region | HIGH |
| Baseline comparison: secondary vs adjacent regions | `secondary_depth_ppm`, `baseline_flux` | Medium |

**Scientific Value:** Visualizes any flux decrement at secondary eclipse phase.

---

### V03 - Duration Consistency
**Purpose:** Check transit duration vs stellar density expectation.

| Plot | Data Source | Priority |
|------|-------------|----------|
| Observed vs expected duration comparison bar chart | `duration_hours`, `expected_duration_hours` | Medium |
| Duration ratio gauge/indicator | `duration_ratio` | Low |
| Context plot: Duration vs period for known planets + this candidate | External context | Low |

**Scientific Value:** Lower priority since this is a single scalar comparison; table display often sufficient.

---

### V04 - Depth Stability
**Purpose:** Check depth consistency across individual transits (variable depth suggests blended EB).

| Plot | Data Source | Priority |
|------|-------------|----------|
| Per-epoch depth time series with error bars | `depths_ppm`, epoch times | **HIGH** |
| Histogram of per-transit depths | `depths_ppm` | HIGH |
| Chi-squared diagnostic: observed vs expected scatter | `depth_scatter_ppm`, `expected_scatter_ppm`, `chi2_reduced` | Medium |
| Dominating epoch highlight (if dom_frac high) | `dominating_epoch_time_btjd`, `dom_frac` | Medium |

**Scientific Value:** Critical for identifying single-event dominated signals or systematic variations.

---

### V05 - V-Shape / Transit Shape
**Purpose:** Distinguish U-shaped (planet) vs V-shaped (grazing EB) transits.

| Plot | Data Source | Priority |
|------|-------------|----------|
| Binned, phase-folded transit with trapezoid model overlay | `tflat_ttotal_ratio`, `t_flat_hours`, `t_total_hours` | **HIGH** |
| tF/tT ratio visualization (ingress/egress vs flat bottom) | `tflat_ttotal_ratio`, `tflat_ttotal_ratio_err` | HIGH |
| Shape diagnostic: bottom depth vs edge depth | `depth_bottom`, `depth_edge` | Medium |

**Scientific Value:** Shape analysis is best understood visually; the trapezoid fit overlay is essential.

---

## Tier 2: Catalog Checks (V06-V07)

### V06 - Nearby EB Search
**Purpose:** Query TESS-EB catalog for EBs that could contaminate the aperture.

| Plot | Data Source | Priority |
|------|-------------|----------|
| Sky map showing target + nearby EB positions | `matches[]` with `ra_deg`, `dec_deg`, `sep_arcsec` | **HIGH** |
| Aperture overlay on DSS/2MASS cutout with EB positions marked | Target coords + `matches[]` | HIGH |
| Period comparison chart: candidate period vs EB periods | `candidate_period_days`, `matches[].period_days`, `delta_1x`, `delta_2x`, `delta_0p5x` | Medium |

**Scientific Value:** Spatial context is crucial for understanding contamination risk.

---

### V07 - ExoFOP TOI Lookup
**Purpose:** Cross-reference with ExoFOP TOI table for disposition and community annotations.

| Plot | Data Source | Priority |
|------|-------------|----------|
| Table/card display (not a plot) | `row` dict fields | Low |
| Historical disposition timeline (if multiple observations) | External TOI history | Low |

**Scientific Value:** Primarily metadata; tabular display is appropriate.

---

## Tier 3: Pixel-Level Checks (V08-V10)

### V08 - Centroid Shift
**Purpose:** Detect if transit source is not the target star via flux-weighted centroid motion.

| Plot | Data Source | Priority |
|------|-------------|----------|
| TPF image with in-transit and out-of-transit centroids marked | `in_transit_centroid_x/y`, `out_of_transit_centroid_x/y` | **HIGH** |
| Vector plot showing centroid shift direction and magnitude | `centroid_shift_pixels`, `centroid_shift_arcsec` | HIGH |
| Centroid time series during transits | Raw centroid data if available | Medium |
| Bootstrap distribution of shift magnitude | `shift_ci_lower_pixels`, `shift_ci_upper_pixels` | Low |

**Scientific Value:** Visual confirmation of centroid motion is compelling evidence for/against blends.

---

### V09 - Difference Image / Pixel-Level Depth Map
**Purpose:** Locate the transit source via per-pixel depth measurements.

| Plot | Data Source | Priority |
|------|-------------|----------|
| 2D heatmap of per-pixel transit depths (ppm) | `depth_map_ppm` from raw | **HIGH** |
| Target pixel vs max-depth pixel overlay on TPF | `target_pixel_row/col`, `max_depth_pixel_row/col` | HIGH |
| In-transit vs out-of-transit difference image | Raw TPF frames | HIGH |
| Concentration ratio visualization | `concentration_ratio`, `distance_to_target_pixels` | Medium |

**Scientific Value:** The depth map is the core diagnostic; essential for blend identification.

---

### V10 - Aperture Dependence
**Purpose:** Check if depth varies with aperture size (contamination indicator).

| Plot | Data Source | Priority |
|------|-------------|----------|
| Depth vs aperture radius curve | `depths_by_aperture_ppm` dict | **HIGH** |
| Aperture growth animation/series on TPF | Aperture radii on TPF stamp | Medium |
| Stability metric gauge | `stability_metric`, `depth_variance_ppm2` | Low |

**Scientific Value:** Depth-aperture curves directly show dilution/contamination effects.

---

## Tier 4: Exovetter Checks (V11-V12)

### V11 - ModShift
**Purpose:** Detect secondary eclipses at arbitrary phases (eccentric EBs).

| Plot | Data Source | Priority |
|------|-------------|----------|
| ModShift periodogram showing primary, secondary, tertiary peaks | `primary_signal`, `secondary_signal`, `tertiary_signal` | **HIGH** |
| Phase-folded LC with ModShift-identified secondary phase marked | Secondary phase from raw | HIGH |
| Signal ratio bar chart | `secondary_primary_ratio`, `tertiary_primary_ratio`, `positive_primary_ratio` | Medium |
| Fred (red noise) diagnostic | `fred`, `false_alarm_threshold` | Low |

**Scientific Value:** The ModShift periodogram is the canonical diagnostic.

---

### V12 - SWEET
**Purpose:** Detect stellar variability mimicking transits.

| Plot | Data Source | Priority |
|------|-------------|----------|
| Sinusoidal fits at P/2, P, 2P overlaid on phase-folded LC | `snr_half_period`, `snr_at_period`, `snr_double_period` | **HIGH** |
| SNR bar chart for each period test | SNR values | HIGH |
| Residuals after sinusoid subtraction | Raw metrics | Medium |

**Scientific Value:** Shows whether variability could explain the observed signal.

---

## Tier 5: False Alarm Checks (V13, V15)

### V13 - Data Gaps
**Purpose:** Detect missing cadences near transits that could cause false alarms.

| Plot | Data Source | Priority |
|------|-------------|----------|
| Transit window coverage plot (per-epoch heatmap) | `worst_epochs[]` with `missing_frac`, `t_center_btjd` | **HIGH** |
| Time series with transit windows and gaps highlighted | LC time, missing regions | HIGH |
| Gap fraction histogram across epochs | `missing_frac` values | Medium |

**Scientific Value:** Gap-edge artifacts are a major TESS false alarm source; visualization is essential.

---

### V15 - Transit Asymmetry
**Purpose:** Detect ramp/step asymmetry around transits (scattered light proxy).

| Plot | Data Source | Priority |
|------|-------------|----------|
| Phase-folded LC showing left vs right of transit | `mu_left`, `mu_right` | **HIGH** |
| Binned flux comparison: pre-transit vs post-transit | Left/right bin means | HIGH |
| Asymmetry sigma gauge | `asymmetry_sigma` | Medium |

**Scientific Value:** Asymmetry visualization directly shows systematic trends.

---

## Tier 6: Extended Checks (V16-V21)

### V16 - Model Competition
**Purpose:** Compare transit model vs alternative models (EB-like, sinusoidal).

| Plot | Data Source | Priority |
|------|-------------|----------|
| BIC/AIC comparison bar chart | `bic_transit_only`, `bic_transit_sinusoid`, `bic_eb_like` | HIGH |
| Multi-model fit overlay on LC | Model fits from raw | **HIGH** |
| Artifact risk breakdown | `artifact_prior_*` metrics | Medium |

**Scientific Value:** Model comparison plots directly address "is this really a planet?"

---

### V17 - Ephemeris Reliability Regime
**Purpose:** Assess how robust the detection is to phase shifts and period variations.

| Plot | Data Source | Priority |
|------|-------------|----------|
| Score vs phase shift (null distribution) | `phase_shift_null` data, `null_percentile` | HIGH |
| Period neighborhood scan | `period_neighborhood_best_score`, peak structure | **HIGH** |
| Top epoch contribution bar chart | `top_contribution_fractions` | Medium |
| Ablation sensitivity plot | `max_ablation_score_drop_fraction` | Medium |

**Scientific Value:** Shows whether detection survives perturbations.

---

### V18 - Ephemeris Sensitivity Sweep
**Purpose:** Test detection stability across preprocessing variants.

| Plot | Data Source | Priority |
|------|-------------|----------|
| Score spread across variants | `metric_variance`, variant results | HIGH |
| Heatmap: detrending x downsampling x outlier policy | Variant grid | Medium |
| Best vs worst variant comparison | `best_variant_id`, `worst_variant_id` | Medium |

**Scientific Value:** Robustness across pipelines builds confidence.

---

### V19 - Alias Diagnostics
**Purpose:** Detect period aliases and harmonics.

| Plot | Data Source | Priority |
|------|-------------|----------|
| Harmonic scores bar chart (P, P/2, 2P, P/3, 3P, etc.) | `base_score_P`, `best_other_score`, harmonic scores | **HIGH** |
| Phase shift event timeline | `phase_shift_events` | Medium |
| Secondary significance at 0.5 phase | `secondary_significance_sigma` | Low |

**Scientific Value:** Alias confusion is common; harmonic comparison is essential.

---

### V20 - Ghost Features
**Purpose:** Detect scattered light / ghost signatures in pixel data.

| Plot | Data Source | Priority |
|------|-------------|----------|
| In-aperture vs out-aperture depth comparison | `in_aperture_depth`, `out_aperture_depth` | **HIGH** |
| Spatial uniformity map of TPF | `spatial_uniformity`, depth gradients | HIGH |
| Ghost risk score breakdown | `ghost_like_score`, `scattered_light_risk`, `aperture_contrast` | Medium |
| Edge gradient visualization | `edge_gradient_strength` | Low |

**Scientific Value:** Ghost/scattered light is a major TESS artifact; pixel context is critical.

---

### V21 - Sector Consistency
**Purpose:** Check depth consistency across multiple sectors.

| Plot | Data Source | Priority |
|------|-------------|----------|
| Per-sector depth with error bars | `sector_measurements[].depth_ppm`, `depth_err_ppm` | **HIGH** |
| Chi-squared consistency diagnostic | `chi2_p_value` | Medium |
| Outlier sector highlight | `outlier_sectors` | Medium |

**Scientific Value:** Multi-sector consistency is strong validation for real signals.

---

## Summary: Priority Matrix

| Priority | Checks |
|----------|--------|
| **HIGH - Essential** | V01, V02, V04, V05, V06, V08, V09, V10, V11, V13, V15, V16, V17, V19, V20, V21 |
| **MEDIUM** | V03, V12, V18 |
| **LOW** | V07 (primarily tabular) |

## Implementation Notes

1. **Phase-folded plots** are needed for: V01, V02, V05, V11, V12, V15
2. **Time-series plots** are needed for: V04, V13
3. **Pixel/image plots** are needed for: V06, V08, V09, V10, V20
4. **Bar/comparison charts** are needed for: V01, V03, V11, V12, V16, V17, V19, V21
5. **Heatmaps** are needed for: V09, V13, V18

## Data Requirements

Most plots can be generated from `CheckResult.metrics` and `CheckResult.raw`:
- `metrics`: Scalar values for annotations and simple charts
- `raw`: Full arrays/dicts for detailed plots (depth maps, epoch lists, etc.)

For pixel checks (V08-V10, V20), the original TPF data should be retained or re-accessible for visualization.
