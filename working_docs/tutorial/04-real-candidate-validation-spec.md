# Tutorial 04: Real Candidate Validation — Specification

## Overview

Create a new tutorial notebook (`docs/tutorials/04-real-candidate-validation.ipynb`) that walks through the complete validation of **TIC 188646744 / TOI-5807.01**, a statistically validated sub-Neptune candidate. The tutorial demonstrates the full vetting workflow using real TESS data and reproduces the results from the technical report.

**Target outcome**: A tutorial that serves as both a learning resource and a reproducible validation report, showing exactly where each number comes from.

---

## Target Values to Reproduce (from technical_report.md)

| Metric | Expected Value | Source Section |
|--------|----------------|----------------|
| SNR (depth/depth_err) | ~26.6 | §5.1 |
| Odd/even Δ | 68.6 ppm (1.73σ) | §5.1 |
| Secondary eclipse | 9.4 ± 8.4 ppm (1.12σ) | §5.1 |
| Transit depth (mean) | ~253 ppm | §5.1 |
| Per-epoch scatter | ~61 ppm | §5.1 |
| FPP (no AO) | 1.9×10⁻² | §5.3 |
| FPP (with PHARO AO) | 1.3×10⁻³ | §5.3 |
| NFPP (with AO) | 2.93×10⁻⁴ | §5.3 |

### Candidate Parameters
- **TIC ID**: 188646744
- **Period**: 14.2423724 days
- **T0**: 3540.26317 BTJD
- **Duration**: 4.56 hours
- **Depth**: 232 ppm
- **Sectors**: 55, 75, 82, 83

### Stellar Parameters
- **Tmag**: 6.88
- **Teff**: 6700 K
- **R★**: 1.738 R☉
- **M★**: 1.43 M☉

---

## Data Sources

### Available in astro-arc-tess working directory

| File | Purpose |
|------|---------|
| `website_plots/stitched_pdcsap.csv` | Stitched light curve (all sectors) |
| `website_plots/sector{55,75,82,83}_pdcsap.csv` | Per-sector light curves |
| `exofop_followup/PHARO_Kcont_plot.tbl` | AO contrast curve (raw) |
| `exofop_followup/PHARO_Kcont_sensitivity.dat` | AO contrast curve (TRICERATOPS format) |
| `comprehensive/tess_pixel_host_disambiguation_summary.json` | Pixel analysis results |
| `notes.md` | Detailed vetting log |

### Data Loading Strategy

**Option A**: Load pre-extracted CSVs from astro-arc-tess (simpler, faster)
- Pro: No MAST queries needed, reproducible offline
- Con: Requires copying data or cross-project reference

**Option B**: Download fresh from MAST using bittr-tess-vetter API
- Pro: Demonstrates real workflow, self-contained
- Con: Network dependency, slower

**Recommendation**: Use Option A for the tutorial (faster, deterministic), but show the MAST download code as a commented alternative.

---

## Tutorial Structure

### Cell 0: Title & Learning Objectives (Markdown)
```markdown
# Real Candidate Validation: TOI-5807.01 (TIC 188646744)

This tutorial demonstrates the complete validation workflow for a real TESS planet candidate.
You will learn:
1. How to load and prepare real TESS light curve data
2. Running the full vetting pipeline on a real candidate
3. Interpreting vetting metrics and comparing to expected values
4. Using contrast curves for FPP calculation
5. How to produce a reproducible validation report
```

### Cell 1: Target Context (Markdown)
- Target summary table (TIC, TOI, stellar params, orbital params)
- Scientific interest (bright host, sub-Neptune, TSM)
- Link to ExoFOP page

### Cell 2: Setup & Imports (Code)
```python
import numpy as np
import pandas as pd
from pathlib import Path

import bittr_tess_vetter.api as btv
from bittr_tess_vetter.api import (
    LightCurve, Ephemeris, Candidate, StellarParams,
    vet_candidate, calculate_fpp,
    odd_even_depth, secondary_eclipse, depth_stability,
    ContrastCurve,
)
```

### Cell 3: Define Candidate Parameters (Code)
```python
# Candidate ephemeris (from ExoFOP / refined fit)
TIC_ID = 188646744
PERIOD_DAYS = 14.2423724
T0_BTJD = 3540.26317
DURATION_HOURS = 4.56
DEPTH_PPM = 232

# Stellar parameters (TIC v8.2)
STELLAR = StellarParams(
    radius_rsun=1.738,
    mass_msun=1.43,
    teff_k=6700,
    logg_cgs=4.1,  # estimated
)

# Create API objects
ephemeris = Ephemeris(
    period_days=PERIOD_DAYS,
    t0_btjd=T0_BTJD,
    duration_hours=DURATION_HOURS,
)
candidate = Candidate(ephemeris=ephemeris, depth_ppm=DEPTH_PPM)
```

### Cell 4: Load Light Curve Data (Code)
- Load stitched PDCSAP from CSV
- Create LightCurve object
- Print summary stats (n_points, time span, sectors)

### Cell 5: Visualize Light Curve (Code)
- Raw time series with transit windows marked
- Phase-folded view with binning
- Optional matplotlib with graceful fallback

### Cell 6: Run Vetting Pipeline (Code)
```python
result = vet_candidate(
    lc, candidate,
    stellar=STELLAR,
    network=False,  # Offline mode for reproducibility
)
```

### Cell 7: Vetting Results Summary (Code + Markdown)
- Table of all check results
- Highlight key metrics vs expected values
- Flag any discrepancies

### Cell 8: Deep Dive: Odd/Even Check (V01) (Code)
- Run individually: `odd_even_depth(lc, candidate)`
- Show detailed metrics
- Compare to report: Δ ≈ 68.6 ppm, 1.73σ

### Cell 9: Deep Dive: Secondary Eclipse (V02) (Code)
- Run individually: `secondary_eclipse(lc, candidate)`
- Show detailed metrics
- Compare to report: 9.4 ± 8.4 ppm, 1.12σ

### Cell 10: Deep Dive: Depth Stability (V04) (Code)
- Run individually: `depth_stability(lc, candidate)`
- Show per-epoch depths
- Compare to report: mean ~253 ppm, scatter ~61 ppm

### Cell 11: Load Contrast Curve (Code)
```python
# Load PHARO AO contrast curve
cc_data = np.loadtxt(contrast_curve_path, delimiter=',', skiprows=9)
contrast_curve = ContrastCurve(
    separation_arcsec=cc_data[:, 0],
    delta_mag=cc_data[:, 1],
    filter='K',
)
```

### Cell 12: Visualize Contrast Curve (Code)
- Plot Δmag vs separation
- Mark key sensitivity points

### Cell 13: FPP Calculation — Baseline (Code)
- Run without contrast curve
- Show FPP breakdown (planet, EB, BEB, NEB, NTP)
- Compare to report: FPP ≈ 1.9×10⁻²

### Cell 14: FPP Calculation — With AO (Code)
- Run with PHARO contrast curve
- Show FPP breakdown
- Compare to report: FPP ≈ 1.3×10⁻³, NFPP ≈ 2.93×10⁻⁴

### Cell 15: Validation Verdict (Markdown + Code)
- Summarize all evidence
- State validation conclusion
- Caveats and limitations

### Cell 16: Pixel Analysis Discussion (Markdown)
- Reference Tutorial 03 for pixel methodology
- Summarize localization findings from technical report
- Note that pixel analysis is "supportive but not decisive"

### Cell 17: Summary & Next Steps (Markdown)
- Bullet summary of what was learned
- Cross-references to other tutorials
- Suggestions for applying to other candidates

---

## Implementation Approach

### Single Agent vs Multiple Agents

**Recommendation: Single agent with iterative development**

Rationale:
1. The tutorial is a single coherent artifact (one notebook)
2. Testing requires running code cells sequentially
3. Values need to match — requires iterative adjustment if API differs
4. Cross-referencing between sections is important

The single agent should:
1. Create the notebook skeleton first
2. Implement cells iteratively, testing each
3. Verify values match the technical report
4. Adjust as needed for API differences

### Potential Challenges

1. **Data path handling**: Need to decide whether to:
   - Copy data into bittr-tess-vetter repo
   - Reference astro-arc-tess via relative path
   - Download from MAST

2. **API differences**: The astro-arc-tess analysis used MCP tools; bittr-tess-vetter API may have slightly different interfaces

3. **FPP calculation**: Requires `PersistentCache` object — need to understand how to set this up for a tutorial context

4. **Value matching**: Numbers may differ slightly due to random seeds, preprocessing, etc. — document acceptable tolerance

### Data Packaging Decision

**Recommendation**: Create a `data/` subdirectory in tutorials with:
- `tic188646744_pdcsap.csv` — stitched light curve
- `pharo_kcont_contrast.dat` — contrast curve

This keeps the tutorial self-contained within bittr-tess-vetter.

---

## Acceptance Criteria

1. **Notebook runs end-to-end** without errors (network=False mode)
2. **Key values match** technical report within reasonable tolerance:
   - Odd/even Δ: 68.6 ± 10 ppm
   - Secondary: 9.4 ± 5 ppm
   - FPP (no AO): within factor of 2
   - FPP (with AO): < 1% (validation threshold met)
3. **Style matches** existing tutorials (imports, structure, presentation)
4. **Reproducible**: Fixed random seeds, documented data sources
5. **Educational**: Clear explanations of what each metric means

---

## Open Questions

1. **Data location**: Copy to bittr-tess-vetter or reference astro-arc-tess?
2. **FPP cache setup**: How does `calculate_fpp` work in tutorial context?
3. **Pixel analysis scope**: Include V08-V10 checks, or just reference Tutorial 03?
4. **MAST fallback**: Include commented code for live data download?

---

## Estimated Effort

- **Research/planning**: Done (this spec)
- **Notebook skeleton**: ~30 min
- **Data preparation**: ~15 min
- **Core implementation**: ~2-3 hours
- **Testing/validation**: ~1-2 hours
- **Polish/documentation**: ~30 min

**Total**: ~4-6 hours of focused work

---

## Next Steps

1. Resolve data location question
2. Investigate `calculate_fpp` API requirements
3. Begin notebook implementation
4. Test iteratively against expected values
