# Missing Citations Spec

## High Priority - DONE ✅

### `src/bittr_tess_vetter/api/periodogram.py`
- ✅ `run_periodogram` - HIPPKE_HELLER_2019_TLS, LOMB_1976, SCARGLE_1982
- ✅ `compute_transit_model` - KOVACS_2002

### `src/bittr_tess_vetter/api/activity.py`
- ✅ `characterize_activity` - MCQUILLAN_2014, DAVENPORT_2016, BASRI_2013
- ✅ `mask_flares` - DAVENPORT_2016, GILBERT_2021

### `src/bittr_tess_vetter/api/recovery.py`
- ✅ `recover_transit` - HIPPKE_2019_WOTAN, HIPPKE_HELLER_2019_TLS, KOVACS_2002
- ✅ `detrend` - HIPPKE_2019_WOTAN

### `src/bittr_tess_vetter/api/aggregation.py`
- ✅ `aggregate_checks` - COUGHLIN_2016, THOMPSON_2018

---

## High Priority - REMAINING

### `src/bittr_tess_vetter/api/pixel_prf.py`
- `build_prf_model` - PRF/PSF construction
- `evaluate_prf_weights` - PRF weighting
- `score_hypotheses_prf_lite` - PRF-based hypothesis scoring
- `score_hypotheses_with_prf` - full PRF hypothesis fitting
- `fit_aperture_hypothesis` - aperture-based hypothesis test
- `predict_depth_vs_aperture` - predicted depth curve from PRF
- `detect_aperture_conflict` - observed vs predicted aperture curves

### `src/bittr_tess_vetter/validation/systematics_proxy.py`
- `compute_systematics_proxy` - red noise / autocorrelation diagnostics

### `src/bittr_tess_vetter/pixel/report.py`
- `generate_pixel_vet_report` - pixel-level vetting diagnostics

---

## Medium Priority (internal implementations behind re-export facades)

### `src/bittr_tess_vetter/transit/vetting.py`
- `split_odd_even` - epoch parity splitting
- `compare_odd_even_depths` - depth comparison statistic
- `compute_odd_even_result` - full odd/even computation

### `src/bittr_tess_vetter/transit/fitting.py`
- `get_ld_coefficients` - limb darkening computation
- `compute_batman_model` - transit model evaluation
- `fit_optimize` - L-BFGS-B transit fitting
- `fit_mcmc` - MCMC sampling
- `compute_derived_parameters` - impact parameter, stellar density
- `fit_transit_model` - main fitting entry point

### `src/bittr_tess_vetter/transit/timing.py`
- `measure_single_transit` - individual transit time measurement
- `measure_all_transit_times` - batch timing
- `compute_ttv_statistics` - TTV analysis

### `src/bittr_tess_vetter/pixel/centroid.py`
- `compute_centroid_shift` - flux-weighted centroid

### `src/bittr_tess_vetter/pixel/difference.py`
- `compute_difference_image` - difference imaging

### `src/bittr_tess_vetter/pixel/aperture.py`
- `compute_aperture_dependence` - aperture contamination

### `src/bittr_tess_vetter/pixel/aperture_family.py`
- `compute_aperture_family_depth_curve` - blend detection via depth vs aperture

### `src/bittr_tess_vetter/pixel/wcs_utils.py`
- `pixel_to_world` - WCS coordinate transform
- `world_to_pixel` - WCS coordinate transform
- `compute_pixel_scale` - WCS-derived scale

---

## No Action Required

- `types.py`, `tolerances.py`, `caps.py`, `canonical.py` - utility only
- `evidence.py` - JSON serialization only
- `transit_masks.py` - basic utility functions
