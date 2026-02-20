# References for tess-vetter API

This document provides full bibliographic entries for the scientific literature
cited in the `lc_only.py` and `transit_primitives.py` modules.

## Core References

### Greisen & Calabretta 2002 (WCS Paper I)

**Bibcode:** `2002A&A...395.1061G`

> Greisen, E.W., & Calabretta, M.R. 2002, "Representations of world coordinates in FITS,"
> A&A, 395, 1061
>
> arXiv: [astro-ph/0207407](https://arxiv.org/abs/astro-ph/0207407)
>
> ADS: [2002A&A...395.1061G](https://ui.adsabs.harvard.edu/abs/2002A%26A...395.1061G)

**Relevance:** Defines the generalized FITS WCS framework (Paper I). This is the
foundational reference for WCS extraction and world↔pixel transforms used in
`pixel/wcs_utils.py` and WCS-aware localization.

---

### Calabretta & Greisen 2002 (WCS Paper II)

**Bibcode:** `2002A&A...395.1077C`

> Calabretta, M.R., & Greisen, E.W. 2002, "Representations of celestial coordinates in FITS,"
> A&A, 395, 1077
>
> arXiv: [astro-ph/0207413](https://arxiv.org/abs/astro-ph/0207413)
>
> ADS: [2002A&A...395.1077C](https://ui.adsabs.harvard.edu/abs/2002A%26A...395.1077C)

**Relevance:** Defines celestial coordinate conventions and map projections within
the FITS WCS framework (Paper II). Used as the conceptual basis for RA/Dec handling
in `pixel/wcs_utils.py`.

---

### Astropy Collaboration 2013

**Bibcode:** `2013A&A...558A..33A`

> Astropy Collaboration 2013, "Astropy: A Community Python Package for Astronomy,"
> A&A, 558, A33
>
> DOI: [10.1051/0004-6361/201322068](https://doi.org/10.1051/0004-6361/201322068)
>
> arXiv: [1307.6212](https://arxiv.org/abs/1307.6212)
>
> ADS: [2013A&A...558A..33A](https://ui.adsabs.harvard.edu/abs/2013A%26A...558A..33A)

**Relevance:** Software reference for `astropy.wcs`, which is the implementation used
for WCS extraction and coordinate transforms in `pixel/wcs_utils.py`.

---

### Kovács, Zucker & Mazeh 2002 (BLS)

**Bibcode:** `2002A&A...391..369K`

> Kovács, G., Zucker, S., & Mazeh, T. 2002, "A box-fitting algorithm in the search for periodic transits,"
> A&A, 391, 369
>
> DOI: [10.1051/0004-6361:20020802](https://doi.org/10.1051/0004-6361:20020802)
>
> arXiv: [astro-ph/0206099](https://arxiv.org/abs/astro-ph/0206099)
>
> ADS: [2002A&A...391..369K](https://ui.adsabs.harvard.edu/abs/2002A%26A...391..369K)

**Relevance:** Foundational Box-fitting Least Squares (BLS) methodology for box-like
transit detection statistics. Forms the methodological lineage for the smooth box
template + weighted least-squares scoring used by MLX diagnostics.

---

### Sundararajan, Taly & Yan 2017 (Integrated Gradients)

**arXiv:** `1703.01365`

> Sundararajan, M., Taly, A., & Yan, Q. 2017, "Axiomatic Attribution for Deep Networks"
>
> arXiv: [1703.01365](https://arxiv.org/abs/1703.01365)

**Relevance:** Defines the Integrated Gradients attribution method, used for MLX
attribution diagnostics (feature importance over the light curve).

---

### Seager & Mallen-Ornelas 2003

**Bibcode:** `2003ApJ...585.1038S`

> Seager, S., & Mallen-Ornelas, G. 2003, "On the Unique Solution of Planet and
> Star Parameters from an Extrasolar Planet Transit Light Curve," ApJ, 585, 1038
>
> DOI: [10.1086/346105](https://doi.org/10.1086/346105)
>
> ADS: [2003ApJ...585.1038S](https://ui.adsabs.harvard.edu/abs/2003ApJ...585.1038S)

**Relevance:** Foundational paper establishing the relationship between transit
duration and stellar density (Equations 3, 9, 19). Introduces transit shape
parameters (tF/tT) for distinguishing planetary transits from grazing eclipses.
Used in V03 (duration_consistency) and V05 (v_shape).

---

### Prsa et al. 2011

**Bibcode:** `2011AJ....141...83P`

> Prsa, A., Batalha, N., Slawson, R.W., et al. 2011, "Kepler Eclipsing Binary
> Stars. I. Catalog and Principal Characterization of 1879 Eclipsing Binaries
> in the First Data Release," AJ, 141, 83
>
> DOI: [10.1088/0004-6256/141/3/83](https://doi.org/10.1088/0004-6256/141/3/83)
>
> ADS: [2011AJ....141...83P](https://ui.adsabs.harvard.edu/abs/2011AJ....141...83P)

**Relevance:** Comprehensive catalog of Kepler eclipsing binaries with
morphology classification. Section 3 provides empirical basis for V-shape vs
U-shape distinction and depth ratio characteristics (EBs show 50-100%
differences between primary and secondary depths). Used in V01 (odd_even_depth)
and V05 (v_shape).

---

### Coughlin & Lopez-Morales 2012

**Bibcode:** `2012AJ....143...39C`

> Coughlin, J.L., & Lopez-Morales, M. 2012, "A Uniform Search for Secondary
> Eclipses of Hot Jupiters in Kepler Q2 Light Curves," AJ, 143, 39
>
> DOI: [10.1088/0004-6256/143/2/39](https://doi.org/10.1088/0004-6256/143/2/39)
>
> ADS: [2012AJ....143...39C](https://ui.adsabs.harvard.edu/abs/2012AJ....143...39C)

**Relevance:** Methodology for detecting secondary eclipses at phase 0.5 in
Kepler light curves. Distinguishes hot Jupiter thermal emission from eclipsing
binary scenarios. Used in V02 (secondary_eclipse).

---

### Fressin et al. 2013

**Bibcode:** `2013ApJ...766...81F`

> Fressin, F., Torres, G., Charbonneau, D., et al. 2013, "The False Positive
> Rate of Kepler and the Occurrence of Planets," ApJ, 766, 81
>
> DOI: [10.1088/0004-637X/766/2/81](https://doi.org/10.1088/0004-637X/766/2/81)
>
> ADS: [2013ApJ...766...81F](https://ui.adsabs.harvard.edu/abs/2013ApJ...766...81F)

**Relevance:** Comprehensive analysis of false positive scenarios for Kepler
planet candidates, including eclipsing binaries and secondary eclipse mimics.
Section 3 describes the false positive taxonomy. Used in V02 (secondary_eclipse).

---

### Coughlin et al. 2016

**Bibcode:** `2016ApJS..224...12C`

> Coughlin, J.L., Mullally, F., Thompson, S.E., et al. 2016, "Planetary
> Candidates Observed by Kepler. VII. The First Fully Uniform Catalog Based
> on the Entire 48-month Data Set (Q1-Q17 DR24)," ApJS, 224, 12
>
> DOI: [10.3847/0067-0049/224/1/12](https://doi.org/10.3847/0067-0049/224/1/12)
>
> ADS: [2016ApJS..224...12C](https://ui.adsabs.harvard.edu/abs/2016ApJS..224...12C)

**Relevance:** Introduces the Kepler Robovetter for automated vetting. Section
4.2 describes the odd/even depth test methodology. This is the foundational
reference for the automated vetting approach used in V01-V05.

---

### Thompson et al. 2018

**Bibcode:** `2018ApJS..235...38T`

> Thompson, S.E., Coughlin, J.L., Hoffman, K., et al. 2018, "Planetary
> Candidates Observed by Kepler. VIII. A Fully Automated Catalog With Measured
> Completeness and Reliability Based on Data Release 25," ApJS, 235, 38
>
> DOI: [10.3847/1538-4365/aab4f9](https://doi.org/10.3847/1538-4365/aab4f9)
>
> ADS: [2018ApJS..235...38T](https://ui.adsabs.harvard.edu/abs/2018ApJS..235...38T)

**Relevance:** Final Kepler DR25 planet candidate catalog with refined
Robovetter. Sections 3.1-3.5 describe the diagnostic tests:
- Section 3.1: Not Transit-Like (V-shape) metric
- Section 3.2: Significant Secondary test
- Section 3.3.1: Odd/even transit depth comparison
- Section 3.4: Planet in Star metric
- Section 3.5: Individual transit metrics

This is the primary reference for all V01-V05 checks.

---

### Twicken et al. 2018

**Bibcode:** `2018PASP..130f4502T`

> Twicken, J.D., Catanzarite, J.H., Clarke, B.D., et al. 2018, "Kepler Data
> Validation I -- Architecture, Diagnostic Tests, and Data Products for
> Vetting Transiting Planet Candidates," PASP, 130, 064502
>
> DOI: [10.1088/1538-3873/aab694](https://doi.org/10.1088/1538-3873/aab694)
>
> ADS: [2018PASP..130f4502T](https://ui.adsabs.harvard.edu/abs/2018PASP..130f4502T)

**Relevance:** Describes the Kepler Data Validation (DV) pipeline diagnostic
tests. Sections 4.3 and 4.5 cover transit duration consistency and depth
stability tests. Used in V03 (duration_consistency) and V04 (depth_stability).

---

### Guerrero et al. 2021

**Bibcode:** `2021ApJS..254...39G`

> Guerrero, N.M., Seager, S., Huang, C.X., et al. 2021, "The TESS Objects of
> Interest Catalog from the TESS Prime Mission," ApJS, 254, 39
>
> DOI: [10.3847/1538-4365/abefe1](https://doi.org/10.3847/1538-4365/abefe1)
>
> ADS: [2021ApJS..254...39G](https://ui.adsabs.harvard.edu/abs/2021ApJS..254...39G)

**Relevance:** TESS TOI catalog and vetting procedures. Section 3.2 describes
the TESS-specific vetting including depth consistency checks. Used in V04
(depth_stability).

---

## Check-to-Reference Mapping

| Check | ID | Primary References |
|-------|----|--------------------|
| Odd/Even Depth | V01 | Coughlin+2016, Thompson+2018, Prsa+2011 |
| Secondary Eclipse | V02 | Coughlin&Lopez-Morales 2012, Thompson+2018, Fressin+2013 |
| Duration Consistency | V03 | Seager&Mallen-Ornelas 2003, Twicken+2018, Thompson+2018 |
| Depth Stability | V04 | Thompson+2018, Twicken+2018, Guerrero+2021 |
| V-Shape | V05 | Seager&Mallen-Ornelas 2003, Prsa+2011, Thompson+2018 |

## Pixel/WCS Reference Notes

Pixel- and WCS-aware utilities in `tess_vetter.pixel.*` are derived from the
Kepler DV / Robovetter diagnostic lineage (difference images, centroid offsets)
and standard FITS WCS conventions (Greisen & Calabretta 2002; Calabretta & Greisen 2002),
implemented via `astropy.wcs` (Astropy Collaboration 2013).

## Adaptation Notes

### odd_even_result (transit_primitives.py)

**Novelty:** adapted

The `odd_even_result` function uses a 10% relative depth difference threshold
instead of the 3-sigma absolute threshold from the original Kepler Robovetter
method (Coughlin et al. 2016, Section 4.2).

**Rationale:** This modification reduces false positives on shallow transits
where photometric noise dominates the absolute depth uncertainty. The relative
threshold is motivated by the empirical observation that real eclipsing binaries
show 50-100% depth differences between primary and secondary (Prsa et al. 2011),
while confirmed planets typically show <5% relative difference.

---

## ADS Query for All References

To retrieve all references in ADS:

```
bibcode:(2003ApJ...585.1038S OR 2011AJ....141...83P OR 2012AJ....143...39C OR 2013ApJ...766...81F OR 2016ApJS..224...12C OR 2018ApJS..235...38T OR 2018PASP..130f4502T OR 2021ApJS..254...39G)
```
