# Tutorial Notebooks Report

## Summary

Created three beginner-friendly tutorial notebooks in `docs/tutorials/` demonstrating key workflows for the `bittr-tess-vetter` library.

## Notebooks Created

### 1. `01-basic-vetting.ipynb`

**Purpose**: Introduces the core vetting workflow for transit candidate validation.

**Content**:
- Introduction to transit vetting concepts (why it matters, types of false positives)
- Importing the API using recommended patterns (`import bittr_tess_vetter.api as btv`)
- Creating synthetic light curve data with numpy (box-shaped transit signal)
- Creating `LightCurve`, `Ephemeris`, and `Candidate` objects
- Running `vet_candidate()` with `network=False`
- Interpreting `VettingBundleResult`:
  - Accessing individual `CheckResult` objects
  - Understanding confidence scores in metrics-only mode
  - Using `get_result()` to access specific checks by ID
- Running specific subsets of checks using the `enabled` parameter
- Common mistakes and error handling examples
- Reference table for LC-only checks (V01-V05)

**Key Teaching Points**:
- The library provides measurements (metrics), not decisions (pass/fail)
- Confidence scores indicate reliability of measurements
- `passed=None` is expected in metrics-only mode

### 2. `02-periodogram-detection.ipynb`

**Purpose**: Demonstrates transit signal detection using periodogram analysis.

**Content**:
- Background on transit detection algorithms (BLS, TLS, Lomb-Scargle)
- Creating synthetic transit data with known parameters
- Using `run_periodogram()` with TLS method
- Understanding `PeriodogramResult`:
  - `best_period`, `best_t0`, `best_duration_hours`
  - SNR and FAP values
  - Multiple peaks and harmonics
- Comparing detected vs true signal parameters
- Phase-folding visualization
- Creating a `Candidate` from periodogram results
- Running vetting on detected candidates
- Comparing TLS (transit) vs LS (sinusoidal) methods
- Periodogram presets (`fast`, `thorough`, `deep`)
- Multi-planet search with `max_planets > 1`

**Key Teaching Points**:
- TLS is for transit detection; LS is for rotation/variability
- Harmonics (P/2, 2P) are common false peaks
- Always verify detections with phase-folded visualization

### 3. `03-pixel-analysis.ipynb`

**Purpose**: Explains pixel-level diagnostics for false positive identification.

**Content**:
- Why pixel analysis matters (background EBs, blends, contamination)
- TESS pixel scale (21 arcsec/pixel)
- Creating synthetic TPF data with Gaussian PSF
- Two scenarios:
  - On-target transit (real planet)
  - Off-target transit (background eclipsing binary)
- Creating `TPFStamp` objects
- Running individual pixel checks:
  - V08 `centroid_shift`: Detects centroid motion during transit
  - V09 `difference_image_localization`: Locates transit source
  - V10 `aperture_dependence`: Checks depth vs aperture size
- Using `vet_pixel()` orchestrator
- Integrating TPF with full `vet_candidate()` pipeline
- Visualization of mean images, difference images, and light curves
- Interpreting results: key indicators of false positives
- Limitations and complementary techniques

**Key Teaching Points**:
- Centroid shift indicates transit source is not the target
- Difference images show where flux loss occurs
- Aperture dependence reveals contamination
- Pixel analysis complements (doesn't replace) other validation methods

## Design Decisions

1. **Synthetic Data Only**: All notebooks use numpy-generated synthetic data to avoid network calls and large file downloads, ensuring fast execution and offline usability.

2. **Execution Speed**: Each notebook should run in under 30 seconds with default parameters.

3. **Progressive Complexity**:
   - Tutorial 01: Simplest workflow (light curve vetting)
   - Tutorial 02: Adds detection step before vetting
   - Tutorial 03: Most complex (3D pixel data)

4. **Astronomy Conventions**: Used BTJD for time, ppm for depth, and standard TESS parameters (2-min cadence, 21 arcsec/pixel).

5. **Metrics-Only Philosophy**: Emphasized that the library provides measurements, not policy decisions, throughout all tutorials.

6. **Optional Visualization**: matplotlib plots are wrapped in try/except blocks so notebooks function without visualization dependencies.

## Limitations and Caveats

### Tutorial 01
- Synthetic transit is a simple box model (real transits have ingress/egress limb darkening)
- Noise is Gaussian (real TESS data has correlated noise and outliers)
- Does not demonstrate stellar parameter integration for V03 (duration_consistency)

### Tutorial 02
- TLS execution time varies significantly with data length and preset
- Multi-planet search may not converge for closely-spaced periods
- Does not demonstrate per-sector analysis (`per_sector=True`)

### Tutorial 03
- Simplified PSF model (real TESS PSF is asymmetric and varies across detector)
- Does not include WCS for coordinate transforms
- Aperture masks are synthetic (real pipeline masks are optimized for SNR)
- Centroid shift detection depends on signal strength and noise level

## Suggestions for Future Notebooks

### Recommended Additional Tutorials

1. **`04-real-data.ipynb`**: Working with real TESS data
   - Downloading light curves from MAST using lightkurve
   - Loading TPF data
   - Handling data quality flags
   - Dealing with sector gaps

2. **`05-catalog-checks.ipynb`**: Network-enabled catalog queries
   - Running V06 (nearby_eb_search) and V07 (exofop_disposition)
   - Providing coordinates and TIC ID
   - Interpreting catalog-based vetting results

3. **`06-transit-fitting.ipynb`**: Advanced transit analysis
   - Using `fit_transit()` for physical parameter estimation
   - TTV analysis with `measure_transit_times()` and `analyze_ttvs()`
   - Transit recovery for active stars

4. **`07-fpp-validation.ipynb`**: Statistical validation
   - Using TRICERATOPS for False Positive Probability
   - Interpreting FPP results
   - Combining vetting checks with statistical validation

5. **`08-batch-vetting.ipynb`**: Processing multiple candidates
   - Efficient batch processing patterns
   - Aggregating results across many targets
   - Export formats for downstream analysis

### Documentation Improvements

- Add a `docs/tutorials/README.md` index file
- Create a quick reference card for all check IDs
- Add troubleshooting guide for common issues

## File Locations

- `/Users/collier/projects/apps/bittr-tess-vetter/docs/tutorials/01-basic-vetting.ipynb`
- `/Users/collier/projects/apps/bittr-tess-vetter/docs/tutorials/02-periodogram-detection.ipynb`
- `/Users/collier/projects/apps/bittr-tess-vetter/docs/tutorials/03-pixel-analysis.ipynb`
