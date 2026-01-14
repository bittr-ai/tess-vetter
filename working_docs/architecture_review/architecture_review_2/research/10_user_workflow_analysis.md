# User Workflow Analysis: bittr-tess-vetter

**Date**: 2026-01-14
**Analyst**: UX Researcher (Claude)
**Package Version**: 0.0.1

---

## Executive Summary

`bittr-tess-vetter` is a domain library for TESS transit candidate vetting. The API is well-designed for researchers familiar with TESS data, but the learning curve is moderate. The library prioritizes **metrics-only** output, delegating policy/pass-fail decisions to downstream applications. This is appropriate for a research library but requires clear documentation.

**Key Findings**:
- Strong: Clean separation between data acquisition and vetting logic
- Strong: Tiered check architecture (LC-only, catalog, pixel, exovetter)
- Friction: FPP requires a `cache` object not easily constructed from the public API
- Friction: No single "TIC ID to vetting results" convenience function
- Friction: Multiple type systems (`LightCurve` facade vs `LightCurveData` internal)

---

## 1. Main Workflow: TIC ID to Vetting Results

### Workflow Overview

A researcher vetting a TESS candidate follows this conceptual path:

```
TIC ID -> Resolve target -> Download light curve -> Define candidate -> Run vetting -> Interpret results
```

### Actual Code Path (Full Workflow)

```python
from bittr_tess_vetter.platform.io.mast_client import MASTClient
from bittr_tess_vetter.api import (
    LightCurve, Ephemeris, Candidate, vet_candidate, run_periodogram
)

# Step 1: Resolve target (if using name instead of TIC ID)
client = MASTClient()
resolved = client.resolve_target("Pi Mensae")
tic_id = resolved.tic_id  # 261136679

# Step 2: Get target info and available sectors
target = client.get_target_info(tic_id)
sectors = client.get_available_sectors(tic_id)

# Step 3: Download light curve(s)
lc_data = client.download_lightcurve(tic_id, sector=sectors[0])

# Step 4: Convert to API types
lc = LightCurve(
    time=lc_data.time,
    flux=lc_data.flux,
    flux_err=lc_data.flux_err,
    quality=lc_data.quality,
    valid_mask=lc_data.valid_mask,
)

# Step 5: Run periodogram to find candidates (or use known ephemeris)
pg_result = run_periodogram(
    time=lc_data.time[lc_data.valid_mask],
    flux=lc_data.flux[lc_data.valid_mask],
    flux_err=lc_data.flux_err[lc_data.valid_mask],
)

# Step 6: Define candidate from detection
eph = Ephemeris(
    period_days=pg_result.best_period,
    t0_btjd=pg_result.best_t0,
    duration_hours=pg_result.best_duration * 24.0,  # Note: days->hours conversion
)
candidate = Candidate(ephemeris=eph, depth_ppm=pg_result.best_depth * 1e6)

# Step 7: Run vetting
result = vet_candidate(
    lc,
    candidate,
    stellar=target.stellar,
    ra_deg=target.ra,
    dec_deg=target.dec,
    tic_id=tic_id,
    network=True,  # Enable catalog checks
)

# Step 8: Interpret results
for r in result.results:
    print(f"{r.id} {r.name}: confidence={r.confidence:.2f}")
print(f"Passed: {result.n_passed}, Failed: {result.n_failed}, Unknown: {result.n_unknown}")
```

**Observation**: This is 8 distinct steps, ~30 lines of code. For a researcher with a known candidate, a simpler path would be valuable.

---

## 2. Minimum Code Path

### If you already have arrays (best case):

```python
from bittr_tess_vetter.api import LightCurve, Ephemeris, Candidate, vet_candidate

# Given: time, flux, flux_err arrays + known ephemeris
lc = LightCurve(time=time, flux=flux, flux_err=flux_err)
eph = Ephemeris(period_days=3.5, t0_btjd=1850.0, duration_hours=2.5)
candidate = Candidate(ephemeris=eph, depth_ppm=500)

result = vet_candidate(lc, candidate)
```

**Lines**: 5 (excluding imports)
**Concepts needed**: `LightCurve`, `Ephemeris`, `Candidate`, `vet_candidate`

### If you have only TIC ID (realistic case):

The library does not provide a single function like:
```python
# Does NOT exist:
result = vet_tic(tic_id=261136679, period=3.5, t0=1850.0, duration_hours=2.5)
```

---

## 3. Friction Points Analysis

### Friction Point 1: Type Conversion Layer

**Problem**: Two light curve types exist:
- `LightCurve` (API facade, in `api.types`)
- `LightCurveData` (internal, in `domain.lightcurve`)

```python
# Download returns internal type
lc_data = client.download_lightcurve(tic_id, sector=1)  # -> LightCurveData

# API functions expect facade type
lc = LightCurve(time=lc_data.time, flux=lc_data.flux, ...)  # Manual conversion

# Or use the conversion method
lc = LightCurve.from_internal(lc_data)  # Exists but not prominently documented
```

**Impact**: Researchers must understand both types and when to convert.

**Recommendation**: Add a convenience method to `MASTClient` or provide a helper:
```python
# Proposed
lc = client.download_as_api_lightcurve(tic_id, sector=1)
```

### Friction Point 2: FPP Requires Cache Object

**Problem**: `calculate_fpp()` requires a `cache` parameter that is not documented or easily constructed:

```python
from bittr_tess_vetter.api import calculate_fpp

# This fails - what is 'cache'?
result = calculate_fpp(
    cache=???,  # PersistentCache type from host application
    tic_id=261136679,
    period=3.5,
    ...
)
```

The type hint shows `cache: PersistentCache = Any`, suggesting this is meant for integration with a host application, not standalone use.

**Impact**: FPP calculation is effectively unavailable to researchers using the library directly.

**Recommendation**: Either:
1. Make cache optional with default in-memory caching
2. Provide clear documentation on constructing a cache object
3. Provide a `calculate_fpp_simple()` that handles caching internally

### Friction Point 3: Duration Units Mismatch

**Problem**: Units are inconsistent across the API:

```python
# Ephemeris uses hours
eph = Ephemeris(duration_hours=2.5)

# PeriodogramResult returns days (best_duration is in days)
pg.best_duration  # 0.104 days

# Conversion required
duration_hours = pg.best_duration * 24.0
```

**Impact**: Easy to introduce bugs by using wrong units.

**Recommendation**: Either standardize on one unit or add explicit `duration_days`/`duration_hours` properties to result objects.

### Friction Point 4: Depth Units Mismatch

**Problem**: Depth can be specified in PPM or fraction:

```python
# Candidate accepts both (with consistency check)
candidate = Candidate(ephemeris=eph, depth_ppm=500)
candidate = Candidate(ephemeris=eph, depth_fraction=0.0005)

# But periodogram returns fraction
pg.best_depth  # 0.0005 (fraction, not ppm)
```

**Impact**: Unit confusion; validation helps but error message is after-the-fact.

### Friction Point 5: Network Checks Are Opt-In

**Problem**: Catalog checks (V06, V07) require `network=True` AND metadata:

```python
# This silently skips catalog checks
result = vet_candidate(lc, candidate)

# This runs catalog checks
result = vet_candidate(
    lc, candidate,
    network=True,
    ra_deg=150.0,
    dec_deg=-30.0,
    tic_id=123456789,
)
```

**Impact**: Researchers may not realize they're missing checks.

**Positive**: Explicit opt-in is good for reproducibility and offline use.

**Recommendation**: Add a warning or summary showing which checks were skipped and why.

---

## 4. Error Handling Quality

### Positive Examples

**Good validation in types**:
```python
eph = Ephemeris(period_days=-1, t0_btjd=1850.0, duration_hours=2.5)
# ValueError: period_days must be positive, got -1
```

**Good shape validation**:
```python
tpf = TPFStamp(time=time_1d, flux=flux_2d, ...)  # Wrong shape
# ValueError: flux must be 3D (n_cadences, n_rows, n_cols), got 2D
```

**Good array length validation**:
```python
lc = LightCurve(time=np.arange(100), flux=np.arange(50), ...)
# ValueError: time and flux must have the same length, got 100 and 50
```

### Error Messages Assessment

| Error Type | Quality | Example |
|------------|---------|---------|
| Type validation | Excellent | Clear message with expected vs actual |
| Shape validation | Excellent | Specific dimension requirements |
| Missing dependencies | Good | `"lightkurve is required. Install with: pip install lightkurve"` |
| Network errors | Good | Wrapped with context about what failed |
| Missing metadata | Good | Check result includes `missing: ["ra_deg"]` in details |

### Error Taxonomy

The library uses a structured error system (`errors.py`):

```python
class ErrorType(str, Enum):
    CACHE_MISS = "CACHE_MISS"
    INVALID_REF = "INVALID_REF"
    INVALID_DATA = "INVALID_DATA"
    INTERNAL_ERROR = "INTERNAL_ERROR"
```

This is useful for programmatic error handling but not currently surfaced to users.

---

## 5. Learning Curve Assessment

### Concepts Required

| Level | Concepts | Example |
|-------|----------|---------|
| Basic | LightCurve, Ephemeris, Candidate | Run basic vetting |
| Intermediate | CheckResult, VettingBundleResult, tiers | Interpret results |
| Advanced | TPFStamp, StellarParams, pixel checks | Full pixel vetting |
| Expert | PRF models, TRICERATOPS, multi-sector | FPP calculation |

### Required Domain Knowledge

1. **TESS data model**: BTJD time system, sectors, TIC IDs
2. **Transit physics**: Period, t0, duration, depth
3. **Vetting concepts**: Odd/even, secondary eclipse, V-shape, centroid
4. **Data formats**: Understanding of normalized flux, quality flags

### Documentation Quality

| Aspect | Rating | Notes |
|--------|--------|-------|
| README quickstart | Good | Working example provided |
| Docstrings | Excellent | Comprehensive with examples |
| Type hints | Excellent | Full typing throughout |
| Inline citations | Excellent | Academic references for algorithms |
| Tutorial/guides | Missing | No step-by-step tutorials |

---

## 6. Convenience Features

### Present Conveniences

1. **Short aliases**:
   ```python
   from bittr_tess_vetter.api import vet  # alias for vet_candidate
   from bittr_tess_vetter.api import periodogram  # alias for run_periodogram
   ```

2. **Auto-conversion in LightCurve**:
   ```python
   lc = LightCurve(time=time, flux=flux)  # flux_err defaults to zeros
   internal = lc.to_internal()  # Handles dtype normalization
   ```

3. **Presets for FPP**:
   ```python
   calculate_fpp(..., preset="fast")  # 50k draws, quick
   calculate_fpp(..., preset="standard")  # 1M draws, thorough
   ```

4. **Performance presets for periodogram**:
   ```python
   run_periodogram(..., preset="fast")
   run_periodogram(..., preset="thorough")
   run_periodogram(..., preset="deep")
   ```

5. **MASTClient helpers**:
   ```python
   client.resolve_target("Pi Mensae")  # Name to TIC ID
   client.get_available_sectors(tic_id)  # List sectors
   client.download_all_sectors(tic_id)  # Batch download
   ```

### Missing Conveniences

1. **No "just vet this TIC ID" function**
2. **No direct light curve to FPP path** (requires cache abstraction)
3. **No multi-sector stitching in main workflow** (exists as `stitch_lightcurves` but separate)
4. **No "known TOI" lookup** (must manually get ephemeris)

---

## 7. Recommended Improvements

### High Priority

1. **Add `vet_tic()` convenience function**:
   ```python
   def vet_tic(
       tic_id: int,
       period_days: float,
       t0_btjd: float,
       duration_hours: float,
       *,
       sectors: list[int] | None = None,
       network: bool = False,
   ) -> VettingBundleResult:
       """One-liner vetting for researchers with known candidates."""
   ```

2. **Document cache construction for FPP** or make it optional

3. **Add explicit warnings when checks are skipped**:
   ```python
   result = vet_candidate(...)
   if result.warnings:
       for w in result.warnings:
           print(f"Warning: {w}")
   ```

### Medium Priority

4. **Standardize units with explicit properties**:
   ```python
   eph.duration_hours  # 2.5
   eph.duration_days   # 0.104 (computed)
   ```

5. **Add `LightCurve.from_mast(tic_id, sector)` class method**

6. **Provide tutorial notebooks** (not just docstring examples)

### Low Priority

7. **Add TOI lookup helper**:
   ```python
   from bittr_tess_vetter.api import get_toi_ephemeris
   eph = get_toi_ephemeris("TOI-123.01")
   ```

8. **Add progress callbacks to long operations** (already present in MASTClient, extend to vetting)

---

## 8. Code Examples for Common Tasks

### Task: Vet a known TOI

```python
from bittr_tess_vetter.platform.io.mast_client import MASTClient
from bittr_tess_vetter.api import LightCurve, Ephemeris, Candidate, vet_candidate

# Known ephemeris for TOI-123
tic_id = 123456789
period, t0, duration = 3.5, 1850.0, 2.5  # From ExoFOP

client = MASTClient()
target = client.get_target_info(tic_id)
lc_data = client.download_lightcurve(tic_id, sector=1)

lc = LightCurve.from_internal(lc_data)
candidate = Candidate(
    ephemeris=Ephemeris(period_days=period, t0_btjd=t0, duration_hours=duration),
    depth_ppm=500,
)

result = vet_candidate(lc, candidate, stellar=target.stellar, network=True,
                       tic_id=tic_id, ra_deg=target.ra, dec_deg=target.dec)
```

### Task: Blind search + vet

```python
from bittr_tess_vetter.platform.io.mast_client import MASTClient
from bittr_tess_vetter.api import LightCurve, Ephemeris, Candidate, run_periodogram, vet_candidate

client = MASTClient()
lc_data = client.download_lightcurve(261136679, sector=1)
valid = lc_data.valid_mask

pg = run_periodogram(
    time=lc_data.time[valid],
    flux=lc_data.flux[valid],
    preset="thorough",
)

if pg.best_period > 0:
    lc = LightCurve.from_internal(lc_data)
    candidate = Candidate(
        ephemeris=Ephemeris(
            period_days=pg.best_period,
            t0_btjd=pg.best_t0,
            duration_hours=pg.best_duration * 24,
        ),
        depth_ppm=pg.best_depth * 1e6,
    )
    result = vet_candidate(lc, candidate)
```

### Task: Run only specific checks

```python
result = vet_candidate(
    lc, candidate,
    enabled={"V01", "V02", "V05"},  # Only odd/even, secondary, v-shape
)
```

### Task: Configure check thresholds

```python
result = vet_candidate(
    lc, candidate,
    config={
        "V01": {"min_transits_per_parity": 3},
        "V04": {"outlier_sigma": 3.0},
    },
)
```

---

## Conclusion

`bittr-tess-vetter` provides a solid foundation for TESS candidate vetting with good API design principles. The main friction points stem from:

1. The gap between data acquisition (`MASTClient`) and the vetting API
2. The FPP cache abstraction that blocks standalone use
3. Lack of a single "TIC ID to vetting" convenience function

The library excels at:
- Clear type safety and validation
- Comprehensive documentation with academic citations
- Flexible configuration for advanced users
- Tiered architecture that works with partial data

For researchers, the learning curve is **moderate** (1-2 hours to become productive, 1-2 days to master). The primary improvements needed are convenience wrappers for common workflows.
