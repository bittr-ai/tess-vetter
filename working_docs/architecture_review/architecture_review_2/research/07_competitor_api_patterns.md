# Competitor API Patterns: Astronomy Python Libraries

**Date:** 2026-01-14
**Scope:** API design patterns from lightkurve, astropy, exoplanet, and transitleastsquares

---

## Executive Summary

Analysis of four major astronomy Python libraries reveals consistent patterns that bittr-tess-vetter should adopt. The most successful libraries share: (1) hierarchical class inheritance with mission-agnostic base classes, (2) method chaining for fluent workflows, (3) explicit subpackage imports with recommended aliases, and (4) layered documentation combining API reference with Jupyter tutorials.

---

## 1. Lightkurve

**Source:** [Lightkurve Documentation](https://docs.lightkurve.org/) | [GitHub](https://github.com/lightkurve/lightkurve) | [DeepWiki Analysis](https://deepwiki.com/lightkurve/lightkurve)

### API Architecture

Lightkurve employs a **layered architecture** with clear separation:

```
Data Access Layer -> Core Data Abstractions -> Analysis Engine -> Visualization Layer
```

### Class Hierarchy Pattern

Mission-specific subclasses inherit from common base classes:

```python
# LightCurve family
TimeSeries (astropy)
  -> LightCurve (base)
      -> KeplerLightCurve
      -> TessLightCurve
      -> FoldedLightCurve

# TargetPixelFile family
TargetPixelFile (base)
  -> KeplerTargetPixelFile
  -> TessTargetPixelFile

# Periodogram family
Periodogram (base)
  -> LombScarglePeriodogram
  -> BoxLeastSquaresPeriodogram
  -> SNRPeriodogram
```

### Key Design Patterns

| Pattern | Implementation | Benefit |
|---------|---------------|---------|
| **Factory** | `detect_filetype()` routes to appropriate readers | Transparent handling without explicit type specification |
| **Strategy** | Correctors (CBV, PLD, Regression) share common interface | Swap algorithms without changing calling code |
| **Template Method** | `RegressionCorrector` defines workflow, subclasses customize | Code reuse with specialization |
| **Observer** | Bokeh widgets trigger cascading updates | Interactive visualization synchronization |

### Method Chaining

Fluent interface enables readable pipelines:

```python
lc = lk.search_lightcurve("TIC 12345").download()
    .remove_outliers().flatten().fold(period=3.5)
    .to_periodogram().plot()
```

### Attribute Resolution

Custom `__getattr__` provides unified access to:
1. Instance attributes
2. Table columns (data)
3. Metadata (FITS headers)

---

## 2. Astropy

**Source:** [Astropy Documentation](https://docs.astropy.org/en/stable/index.html) | [Importing Guide](https://docs.astropy.org/en/stable/importing_astropy.html)

### Subpackage Organization

Astropy uses **domain-driven subpackages**:

| Subpackage | Purpose | Key Classes |
|------------|---------|-------------|
| `astropy.units` | Physical quantities | `Unit`, `Quantity` |
| `astropy.coordinates` | Celestial positions | `SkyCoord`, `Angle` |
| `astropy.io.fits` | FITS file handling | `HDUList`, `Header` |
| `astropy.time` | Time representations | `Time`, `TimeDelta` |
| `astropy.table` | Tabular data | `Table`, `QTable` |
| `astropy.timeseries` | Time series data | `TimeSeries`, `BinnedTimeSeries` |
| `astropy.cosmology` | Cosmological models | `WMAP7`, `Planck18` |
| `astropy.wcs` | World coordinates | `WCS` |

### Import Conventions

Astropy establishes **recommended aliases**:

```python
from astropy import units as u
from astropy import coordinates as coord
from astropy.io import fits
from astropy.table import Table
from astropy.cosmology import WMAP7
```

**Critical rule:** Never use `from astropy import *`

### Design Philosophy

> "Code using astropy should result in concise and easily readable code, even by those new to Python. Typical operations should appear in code similar to how they would appear if expressed in spoken or written language."

### Affiliated Package Ecosystem

Astropy distinguishes:
- **Core**: `astropy` package itself
- **Coordinated**: Maintained by Astropy Project (e.g., `astropy-healpix`)
- **Affiliated**: Community packages following Astropy standards (e.g., `lightkurve`)

---

## 3. Exoplanet (PyMC)

**Source:** [Exoplanet Docs](https://docs.exoplanet.codes/) | [API Reference](https://docs.exoplanet.codes/en/latest/user/api/)

### Module Structure

Exoplanet organizes by **functional domain**:

```python
exoplanet/
  orbits/           # Orbital mechanics
    - KeplerianOrbit
    - TTVOrbit
    - SimpleTransitOrbit
  light_curves/     # Transit modeling
    - LimbDarkLightCurve
    - SecondaryEclipseLightCurve
  distributions/    # Bayesian priors
    - QuadLimbDark
    - impact_parameter
    - eccentricity.kipping13()
  estimators/       # Period finding
    - lomb_scargle_estimator
    - bls_estimator
```

### Composable Design

Exoplanet favors **composition over inheritance**:

```python
orbit = xo.orbits.KeplerianOrbit(period=3.5, t0=0.0, b=0.5)
lc_model = xo.LimbDarkLightCurve(u1=0.3, u2=0.2)
light_curve = lc_model.get_light_curve(orbit=orbit, r=0.1, t=times)
```

### PyMC Integration

Domain objects integrate directly with probabilistic programming:

```python
with pm.Model():
    # Prior distributions from exoplanet
    u = xo.distributions.QuadLimbDark("u")
    b = xo.distributions.impact_parameter("b", ror=0.1)

    # Physical model
    orbit = xo.orbits.KeplerianOrbit(period=period, t0=t0, b=b)
    lc = xo.LimbDarkLightCurve(u[0], u[1]).get_light_curve(orbit=orbit, r=r)
```

---

## 4. Transit Least Squares

**Source:** [TLS Documentation](https://transitleastsquares.readthedocs.io/en/latest/Python%20interface.html) | [PyPI](https://pypi.org/project/transitleastsquares/)

### Minimal API Surface

TLS provides an intentionally **simple interface**:

```python
from transitleastsquares import transitleastsquares

model = transitleastsquares(time, flux)
results = model.power(
    period_min=0.5,
    period_max=20.0,
    R_star=1.0,
    M_star=1.0
)
```

### Rich Return Object

Single method returns comprehensive results:

```python
results.period          # Best-fit period
results.T0              # First mid-transit time
results.duration        # Transit duration
results.depth           # Transit depth
results.SDE             # Signal Detection Efficiency
results.FAP             # False Alarm Probability
results.snr             # Signal-to-noise ratio
results.odd_even_mismatch  # EB indicator
results.model_folded_model # Phase-folded model for plotting
```

### Utility Functions

Separate functions for common tasks:

```python
from transitleastsquares import catalog_info, transit_mask, cleaned_array

# Stellar priors from mission catalogs
ab, mass, radius = catalog_info(TIC_ID=12345)

# Boolean mask for in-transit points
mask = transit_mask(time, period=3.5, duration=0.1, T0=0.0)

# Data cleaning
t_clean, y_clean = cleaned_array(time, flux)
```

---

## 5. Common Patterns Across Libraries

### 5.1 Import Conventions

| Library | Pattern | Example |
|---------|---------|---------|
| Astropy | Subpackage aliases | `from astropy import units as u` |
| Lightkurve | Direct package import | `import lightkurve as lk` |
| Exoplanet | Namespace shorthand | `import exoplanet as xo` |
| TLS | Class import | `from transitleastsquares import transitleastsquares` |

### 5.2 Data Container Design

All libraries use **immutable-ish data containers** with:
- Guaranteed required fields (time, flux, flux_err)
- Optional metadata access
- Conversion methods (`to_periodogram()`, `to_lightcurve()`)
- Copy-on-modify semantics

### 5.3 Documentation Structure

| Component | Purpose | Examples |
|-----------|---------|----------|
| **Quickstart** | 5-minute onboarding | `lk.search_lightcurve().download().plot()` |
| **Tutorials** | Jupyter notebooks by topic | "Recover a planet", "Detrending light curves" |
| **API Reference** | Sphinx autodoc | Class/method documentation |
| **How-to Guides** | Task-oriented recipes | "How to mask flares" |

### 5.4 Notebook Organization

Lightkurve's numbered tutorial structure:

```
tutorials/
  1-getting-started/
    1.01-what-are-target-pixel-files.ipynb
    1.02-what-are-lightcurves.ipynb
  2-creating-light-curves/
    2.01-how-to-detrend.ipynb
    2.02-recover-a-planet.ipynb
  3-science-examples/
    3.01-asteroseismology.ipynb
```

**Key practice:** Notebooks stored without output; rendered versions hosted separately.

---

## 6. Recommendations for bittr-tess-vetter

### 6.1 Patterns to Adopt

| Pattern | Current State | Recommendation |
|---------|--------------|----------------|
| **Recommended alias** | None documented | Document `import bittr_tess_vetter.api as btv` |
| **Method chaining** | Not used | Add `Candidate.with_lightcurve().with_tpf()` builders |
| **Factory methods** | Partial | Add `Candidate.from_mast(tic_id)` factory |
| **Rich result objects** | `CheckResult` exists | Ensure consistent structure across all checks |
| **Subpackage imports** | Mixed | Standardize `from bittr_tess_vetter.api import ...` |

### 6.2 Naming Conventions

Following astropy's domain-driven approach:

| Current | Recommended | Rationale |
|---------|-------------|-----------|
| `catalog.py` | `checks_catalog.py` | Clarify purpose (vetting checks) |
| `catalogs.py` | Keep as-is | Platform-level catalog clients |
| `vet_lc_only` | `run_lc_checks` | More explicit action verb |

### 6.3 Documentation Structure

Adopt lightkurve's tiered approach:

```
docs/
  quickstart.md           # 5-minute getting started
  tutorials/
    01-basic-vetting.ipynb
    02-custom-checks.ipynb
    03-batch-processing.ipynb
  api/
    index.rst             # Sphinx autodoc
  howto/
    mask-flares.md
    add-custom-check.md
```

### 6.4 Patterns to Avoid

| Anti-Pattern | Seen In | Risk |
|--------------|---------|------|
| `from package import *` | Legacy code | Namespace pollution |
| Deep inheritance hierarchies | Some older libs | Fragile base class problem |
| Stateful singletons | Configuration objects | Testing complexity |
| Mixed sync/async in same interface | MAST clients | Confusing API surface |

### 6.5 Class Hierarchy Suggestion

Inspired by lightkurve's mission-agnostic base classes:

```python
# Current (implicit)
Candidate -> CheckResult

# Recommended (explicit)
VettingTarget (base)
  -> TESSCandidate
  -> KeplerCandidate (future)

VettingResult (base)
  -> CheckResult
  -> TransitFitResult
  -> TTVResult
```

---

## 7. Key Takeaways

1. **Simple entry points, rich results**: TLS shows that a single `power()` method returning 30+ attributes is preferable to 30 separate methods.

2. **Composition over inheritance**: Exoplanet's design of passing `orbit` objects to `LimbDarkLightCurve.get_light_curve()` enables flexible composition.

3. **Explicit imports with aliases**: Astropy's `from astropy import units as u` pattern reduces namespace conflicts while keeping code readable.

4. **Layered documentation**: All successful libraries provide quickstart + tutorials + API reference, never just one.

5. **Factory methods for data access**: Lightkurve's `search_lightcurve().download()` pattern is universally adopted because it abstracts data source complexity.

---

## Sources

- [Lightkurve Documentation](https://docs.lightkurve.org/)
- [Lightkurve GitHub](https://github.com/lightkurve/lightkurve)
- [Lightkurve DeepWiki Analysis](https://deepwiki.com/lightkurve/lightkurve)
- [Astropy Documentation](https://docs.astropy.org/en/stable/index.html)
- [Astropy Import Guide](https://docs.astropy.org/en/stable/importing_astropy.html)
- [Astropy A&A Paper](https://www.aanda.org/articles/aa/full_html/2013/10/aa22068-13/aa22068-13.html)
- [Exoplanet Documentation](https://docs.exoplanet.codes/)
- [Exoplanet API Reference](https://docs.exoplanet.codes/en/latest/user/api/)
- [Transit Least Squares Documentation](https://transitleastsquares.readthedocs.io/en/latest/Python%20interface.html)
- [Astropy Documentation Guidelines](https://docs.astropy.org/en/latest/development/docguide.html)
- [Lightkurve Tutorials Structure](https://github.com/lightkurve/lightkurve/tree/main/docs/source/tutorials)
