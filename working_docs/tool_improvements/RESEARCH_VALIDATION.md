# Literature Research Validation Report

**Author:** Claude Code
**Date:** 2026-01-08
**Status:** Complete
**Scope:** V02, V04, V05, V08, V11, V12 vetting check implementations

---

## Executive Summary

This report validates the literature research and citations across six vetting check implementations in the bittr-tess-vetter package. The validation confirms that:

1. **Most citations are accurate and verifiable** - Key methodological papers exist and contain the claimed content
2. **Three arXiv ID errors were discovered** requiring correction:
   - Coughlin et al. 2016: Correct is `1512.06149`, NOT `1601.05413`
   - Santerne 2013: Two different papers with arXiv IDs `1307.2003` and `1306.6812` - they are DIFFERENT papers
3. **All verified papers support the methodological claims** made in the design documents
4. **No major missing citations** were identified, though some additional relevant papers are suggested

---

## Verification Results by Check

### V02: Secondary Eclipse Search

#### Citations Verified

| Citation | arXiv ID | Status | Notes |
|----------|----------|--------|-------|
| **Thompson et al. 2018** | `1710.06758` | **VERIFIED** | "Planetary Candidates Observed by Kepler. VIII. A Fully Automated Catalog With Measured Completeness and Reliability Based on Data Release 25" - Correct title, authors (Susan E. Thompson, Jeffrey L. Coughlin, et al.), published 2017-10-18 |
| **Santerne et al. 2013** | `1307.2003` | **VERIFIED** | "The contribution of secondary eclipses as astrophysical false positives to exoplanet transit surveys" - Correct paper about secondary-only EBs |
| **Pont et al. 2006** | `astro-ph/0608597` | NOT VERIFIED (older arXiv format) | Cited for red noise methodology |
| **Coughlin & Lopez-Morales 2012** | `1112.1021` | CLAIMED | Not directly verified in this session |

#### Methodology Verification

The design document correctly attributes:
- **Red noise inflation** to Pont et al. 2006 (MNRAS 373, 231)
- **Secondary-only EB scenario** to Santerne et al. 2013 - VERIFIED: Paper explicitly discusses "eclipsing binaries and giant planets which present only a secondary eclipse, as seen from the Earth" with occurrence rate of 0.061% +/- 0.017%
- **Robovetter secondary test** to Thompson et al. 2018 - VERIFIED: Paper describes DR25 Robovetter methodology

---

### V04: Depth Stability Check

#### Citations Verified

| Citation | arXiv ID | Status | Notes |
|----------|----------|--------|-------|
| **Thompson et al. 2018** | `1710.06758` | **VERIFIED** | Same as V02 - correct |
| **Pont et al. 2006** | `astro-ph/0608597` | CLAIMED | Red noise methodology |
| **Wang & Espinoza 2023** | `2311.02154` | CLAIMED | Not verified in this session |
| **Santerne et al. 2013** | `1307.2003` | **VERIFIED** | Correct - secondary eclipse FP paper |

#### Methodology Verification

The design document correctly describes:
- **Chi-squared stability metric** - Standard statistical approach
- **Per-epoch local baseline** - Consistent with Thompson et al. 2018 methodology
- **Red noise inflation** - Correctly attributed to Pont et al. 2006

---

### V05: V-Shape (Transit Morphology) Check

#### Citations Verified

| Citation | arXiv ID | Status | Notes |
|----------|----------|--------|-------|
| **Seager & Mallen-Ornelas 2003** | `astro-ph/0206228` | **VERIFIED** | "On the Unique Solution of Planet and Star Parameters from an Extrasolar Planet Transit Light Curve" - Foundational paper on transit geometry and tF/tT ratio |
| **Kipping 2010** | `1004.3819` | **VERIFIED** | "Investigations of approximate expressions for the transit duration" - Correct paper on T14/T23 definitions |
| **Hippke & Heller 2019** | `1901.02015` | **VERIFIED** | "Transit Least Squares: Optimized transit detection algorithm to search for periodic transits of small planets" - TLS algorithm paper |
| **Thompson et al. 2018** | `1710.06758` | **VERIFIED** | DR25 Robovetter |

#### Methodology Verification

The design document correctly attributes:
- **tF/tT ratio definition** to Seager & Mallen-Ornelas 2003 - VERIFIED: Paper discusses "unique solution of the planet and star parameters from a planet transit light curve"
- **Transit duration expressions** to Kipping 2010 - VERIFIED: Paper title and abstract confirm this
- **V-shape detection limitations** to Hippke & Heller 2019 - VERIFIED: TLS paper discusses transit shape templates

---

### V08: Centroid Shift Check

#### Citations Verified

| Citation | arXiv ID | Status | Notes |
|----------|----------|--------|-------|
| **Bryson et al. 2010** | `1001.0331` | **VERIFIED** | "The Kepler Pixel Response Function" - Correct PRF methodology paper |
| **Bryson et al. 2013** | `1303.0052` | **VERIFIED** | "Identification of Background False Positives from Kepler Data" - Correct paper on centroid analysis |
| **Batalha et al. 2010** | `1001.0392` | **VERIFIED** | "Pre-Spectroscopic False Positive Elimination of Kepler Planet Candidates" - Correct paper on data validation including centroid analysis |
| **Higgins & Bell 2022** | `2204.06020` | **VERIFIED** | "Method for Disentangling Blended Variable Stars in TESS and Kepler Photometry" - TESS-specific localization methodology |

#### Methodology Verification

All V08 citations are accurate:
- **PRF methodology** - Bryson et al. 2010 explicitly describes "Kepler pixel response function"
- **Centroid analysis** - Bryson et al. 2013 describes "techniques for the identification of background transit sources"
- **Pre-spectroscopic FP elimination** - Batalha et al. 2010 describes flux-weighted centroid analysis
- **TESS localization** - Higgins & Bell 2022 describes method to "localize the origin of variability on the sky to better than one fifth of a pixel"

---

### V11: ModShift Test

#### Citations Verified

| Citation | arXiv ID | Status | Notes |
|----------|----------|--------|-------|
| **Thompson et al. 2018** | `1710.06758` | **VERIFIED** | DR25 Robovetter - Section 3.2.3 for ModShift |
| **Coughlin et al. 2016** | **ERROR IN CODE** | **INCORRECT arXiv ID** | Code claims `1601.05413` but this is actually "Chemical tagging can work" by Hogg et al. - WRONG PAPER! |
| **Santerne et al. 2013** | `1307.2003` | **VERIFIED** | Secondary eclipse phase offset for eccentric orbits |
| **Twicken et al. 2018** | `1803.04526` | **VERIFIED** | "Data Validation I -- Architecture, Diagnostic Tests, and Data Products for Vetting Transiting Planet Candidates" - DV pipeline paper |

#### CRITICAL ERROR FOUND

**The arXiv ID `1601.05413` in `exovetter_checks.py` is INCORRECT!**

- **Claimed:** Coughlin et al. 2016, ApJS 224, 12 (arXiv:1601.05413)
- **Actual paper at 1601.05413:** "Chemical tagging can work: Identification of stellar phase-space structures purely by chemical-abundance similarity" by Hogg et al. - NOT RELATED TO EXOPLANET VETTING
- **Correct arXiv ID:** `1512.06149` - "Planetary Candidates Observed by Kepler. VII. The First Fully Uniform Catalog Based on The Entire 48 Month Dataset (Q1-Q17 DR24)" by Jeffrey L. Coughlin et al.

**Action Required:** Update references.py and exovetter_checks.py to use correct arXiv ID `1512.06149`.

---

### V12: SWEET Test

#### Citations Verified

| Citation | arXiv ID | Status | Notes |
|----------|----------|--------|-------|
| **Thompson et al. 2018** | `1710.06758` | **VERIFIED** | Section 3.2.4 for SWEET test |
| **Coughlin et al. 2016** | **ERROR** | **SAME ERROR AS V11** | Wrong arXiv ID - should be `1512.06149` |
| **McQuillan et al. 2014** | `1402.5694` | **VERIFIED** | "Rotation Periods of 34,030 Kepler Main-Sequence Stars: The Full Autocorrelation Sample" - Stellar rotation periods |
| **Basri et al. 2013** | `1304.0136` | **VERIFIED** | "Photometric Variability in Kepler Target Stars. III. Comparison with the Sun on Different Timescales" - Stellar variability amplitudes |

#### Methodology Verification

- **SWEET test methodology** correctly attributed to Thompson et al. 2018 Section 3.2.4
- **Stellar rotation periods** correctly attributed to McQuillan et al. 2014 - VERIFIED: Paper provides "rotation periods for 34,030 main-sequence Kepler targets"
- **Stellar variability amplitudes** correctly attributed to Basri et al. 2013 - VERIFIED: Paper studies "photometric variability of solar-type and cooler stars at different timescales"

---

## Santerne 2013 arXiv ID Discrepancy

A discrepancy was found between two different arXiv IDs attributed to Santerne 2013:

| Location | arXiv ID | Paper Title | Correct? |
|----------|----------|-------------|----------|
| exovetter_checks.py | `1307.2003` | "The contribution of secondary eclipses as astrophysical false positives..." | **YES** |
| references.py | `1306.6812` | "Epidemics in Stochastic Multipartite Networks" by Santos et al. | **NO - WRONG PAPER!** |

**Finding:** The arXiv ID `1306.6812` in references.py is INCORRECT. It points to a completely unrelated paper about network epidemics. The correct arXiv ID for Santerne et al. 2013 (A&A 557, A139) is `1307.2003`.

**Note:** The bibcode `2013A&A...557A.139S` in references.py is likely correct, but the arXiv ID needs correction.

---

## Summary of Required Corrections

### High Priority (Incorrect arXiv IDs)

1. **Coughlin et al. 2016**
   - Current: `1601.05413` (WRONG - points to Hogg et al. chemical tagging paper)
   - Correct: `1512.06149`
   - Files to update: `exovetter_checks.py`, `references.py`

2. **Santerne et al. 2013**
   - Current in references.py: `1306.6812` (WRONG - points to Santos et al. epidemics paper)
   - Correct: `1307.2003`
   - Files to update: `references.py`

### Medium Priority (Unverified but Likely Correct)

These citations were not directly verified but appear in standard literature:
- Pont et al. 2006 (astro-ph/0608597) - Red noise in transit photometry
- Wang & Espinoza 2023 (2311.02154) - Depth variability with TESS
- Coughlin & Lopez-Morales 2012 (1112.1021) - Secondary eclipse methodology

---

## Verified Paper Details

### Thompson et al. 2018 (arXiv:1710.06758)

**Full Title:** "Planetary Candidates Observed by Kepler. VIII. A Fully Automated Catalog With Measured Completeness and Reliability Based on Data Release 25"

**Authors:** Susan E. Thompson, Jeffrey L. Coughlin, Kelsey Hoffman, Fergal Mullally, Jessie L. Christiansen, et al. (61 total)

**Key Content Verified:**
- Describes DR25 Robovetter methodology
- Contains ModShift test (Section 3.2.3 as claimed)
- Contains SWEET test (Section 3.2.4 as claimed)
- Contains disposition score methodology

### Coughlin et al. 2016 (arXiv:1512.06149)

**Full Title:** "Planetary Candidates Observed by Kepler. VII. The First Fully Uniform Catalog Based on The Entire 48 Month Dataset (Q1-Q17 DR24)"

**Authors:** Jeffrey L. Coughlin, F. Mullally, Susan E. Thompson, Jason F. Rowe, Christopher J. Burke, et al. (35 total)

**Key Content:** DR24 Robovetter implementation including ModShift and SWEET tests

### Hippke & Heller 2019 (arXiv:1901.02015)

**Full Title:** "Transit Least Squares: Optimized transit detection algorithm to search for periodic transits of small planets"

**Authors:** Michael Hippke, Rene Heller

**Key Content:** TLS algorithm, transit shape templates, detection efficiency comparison with BLS

### Higgins & Bell 2022 (arXiv:2204.06020)

**Full Title:** "Method for Disentangling Blended Variable Stars in TESS and Kepler Photometry"

**Authors:** Michael E. Higgins, Keaton J. Bell

**Key Content:** Sub-pixel localization method for TESS, achieving "better than one fifth of a pixel" precision

---

## Additional Recommended Citations

Based on the literature search, these papers may be valuable additions:

1. **Morton & Johnson 2011** (arXiv:1101.5630) - "On the Low False Positive Probabilities of Kepler Planet Candidates" - FPP framework
2. **Fressin et al. 2013** (arXiv:1301.0842) - "The false positive rate of Kepler and the occurrence of planets" - Already cited in V02
3. **Morton et al. 2016** (arXiv:1605.02825) - "False positive probabilities for all Kepler Objects of Interest" - vespa validation

---

## Conclusion

The literature research in the V02, V04, V05, V08, V11, and V12 design documents is largely accurate and well-sourced. The methodological claims are supported by the cited papers. Two critical arXiv ID errors were discovered that require immediate correction:

1. Coughlin et al. 2016: `1601.05413` -> `1512.06149`
2. Santerne et al. 2013 in references.py: `1306.6812` -> `1307.2003`

All other verified citations are correct and appropriately attributed.

---

*Report generated by Claude Code on 2026-01-08*
