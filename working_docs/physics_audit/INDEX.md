# Physics Audit Index

This is the ordered queue for the function-by-function audit.

## 0) Setup

- `CONVENTIONS.md` (units, time bases, definitions)
- `REVIEW_TEMPLATE.md` (what we check for each function)
- `INVENTORY.md` (API → implementation map; entry points used by agents first)
- `NEXT_PRIORITIES.md` (dynamic “what to audit next” queue based on blast radius)
- Module notes live in `modules/in_progress/` while active and move to `modules/completed/` when all checkboxes are checked.

## 1) Early-agent entry points (review first)

### Types + contracts (prevents silent physics errors)
- ✅ `src/bittr_tess_vetter/api/types.py` → `working_docs/physics_audit/modules/completed/api_types.md`
- ✅ `src/bittr_tess_vetter/api/lightcurve.py` → `working_docs/physics_audit/modules/completed/api_lightcurve.md`

### Light curve assembly / normalization
- ✅ `src/bittr_tess_vetter/api/stitch.py` → `working_docs/physics_audit/modules/completed/api_stitch.md`

### Transit masking primitives (used everywhere)
- ✅ `src/bittr_tess_vetter/api/compute_transit.py` → `working_docs/physics_audit/modules/completed/api_compute_transit.md`
- ✅ `src/bittr_tess_vetter/api/transit_masks.py` → `working_docs/physics_audit/modules/completed/api_transit_masks.md`
- ✅ `src/bittr_tess_vetter/validation/base.py` → `working_docs/physics_audit/modules/completed/api_validation_base_masks.md`

### Detection (periodogram) + minimal models
- ✅ `src/bittr_tess_vetter/api/periodogram.py` → `working_docs/physics_audit/modules/completed/api_periodogram.md`
- ✅ `src/bittr_tess_vetter/api/transit_model.py` → `working_docs/physics_audit/modules/completed/api_transit_model.md`

### LC-only vetting (fast triage)
- ✅ `src/bittr_tess_vetter/api/lc_only.py` → `working_docs/physics_audit/modules/completed/api_lc_only.md`

## 2) Next layer (review after early-agent surface is solid)

- ✅ `src/bittr_tess_vetter/api/recovery.py` → `working_docs/physics_audit/modules/completed/api_recovery.md`
- ✅ `src/bittr_tess_vetter/api/timing.py` → `working_docs/physics_audit/modules/completed/api_timing.md`
- ✅ `src/bittr_tess_vetter/api/activity.py` → `working_docs/physics_audit/modules/completed/api_activity.md`
- ✅ `src/bittr_tess_vetter/api/fpp.py` → `working_docs/physics_audit/modules/completed/api_fpp.md`
- ✅ `src/bittr_tess_vetter/api/pixel.py` / `src/bittr_tess_vetter/api/wcs_localization.py` → `working_docs/physics_audit/modules/completed/api_wcs_localization.md`

## 3) Deep pixel + PRF + advanced inference (later)

- ✅ `src/bittr_tess_vetter/api/pixel_prf.py` → `working_docs/physics_audit/modules/completed/api_pixel_prf.md`
- ✅ `src/bittr_tess_vetter/api/transit_fit.py` (+ `src/bittr_tess_vetter/transit/batman_model.py`) → `working_docs/physics_audit/modules/completed/api_transit_fit.md`
- ✅ `src/bittr_tess_vetter/validation/exovetter_checks.py` → `working_docs/physics_audit/modules/completed/validation_exovetter_checks.md`
- ✅ `src/bittr_tess_vetter/validation/lc_checks.py` → `working_docs/physics_audit/modules/completed/validation_lc_checks.md` (V01–V05 audited; extend later for V06–V10 as needed)

## 4) Remaining astrophysics surface (not yet audited)

Auto-discovered by scanning `src/bittr_tess_vetter/**/*.py` and subtracting the ✅ list above (excluding `__init__.py` files).

### 4.1) Public API wrappers / entry points

- ✅ `src/bittr_tess_vetter/api/aperture.py` → `working_docs/physics_audit/modules/completed/api_aperture_dependence.md`
- ✅ `src/bittr_tess_vetter/api/aperture_family.py` → `working_docs/physics_audit/modules/completed/api_aperture_family.md`
- ✅ `src/bittr_tess_vetter/api/bls_like_search.py` → `working_docs/physics_audit/modules/completed/api_bls_like_search.md`
- ☐ `src/bittr_tess_vetter/api/canonical.py`
- ☐ `src/bittr_tess_vetter/api/caps.py`
- ✅ `src/bittr_tess_vetter/api/catalog.py` → `working_docs/physics_audit/modules/completed/api_catalog_and_catalogs.md`
- ✅ `src/bittr_tess_vetter/api/catalogs.py` → `working_docs/physics_audit/modules/completed/api_catalog_and_catalogs.md`
- ✅ `src/bittr_tess_vetter/api/centroid.py` → `working_docs/physics_audit/modules/completed/api_centroid_shift.md`
- ✅ `src/bittr_tess_vetter/api/detection.py` → `working_docs/physics_audit/modules/completed/api_detection_and_detrend.md`
- ✅ `src/bittr_tess_vetter/api/detrend.py` → `working_docs/physics_audit/modules/completed/api_detection_and_detrend.md`
- ☐ `src/bittr_tess_vetter/api/difference.py`
- ✅ `src/bittr_tess_vetter/api/ephemeris_specificity.py` → `working_docs/physics_audit/modules/completed/validation_ephemeris_specificity_and_prefilter.md`
- ✅ `src/bittr_tess_vetter/api/evidence.py` → `working_docs/physics_audit/modules/completed/api_evidence_and_report.md`
- ✅ `src/bittr_tess_vetter/api/exovetter.py` → `working_docs/physics_audit/modules/completed/api_exovetter.md`
- ✅ `src/bittr_tess_vetter/api/io.py` → `working_docs/physics_audit/modules/completed/api_io_and_target.md`
- ✅ `src/bittr_tess_vetter/api/joint_inference.py` → `working_docs/physics_audit/modules/completed/api_joint_inference.md`
- ✅ `src/bittr_tess_vetter/api/localization.py` → `working_docs/physics_audit/modules/completed/api_proxy_localization.md`
- ✅ `src/bittr_tess_vetter/api/mlx.py` → `working_docs/physics_audit/modules/completed/api_mlx_and_compute_mlx_detection.md`
- ✅ `src/bittr_tess_vetter/api/model_competition.py` → `working_docs/physics_audit/modules/completed/model_competition.md`
- ✅ `src/bittr_tess_vetter/api/pixel_host_hypotheses.py` → `working_docs/physics_audit/modules/completed/api_pixel_prf.md`
- ✅ `src/bittr_tess_vetter/api/prefilter.py` → `working_docs/physics_audit/modules/completed/validation_ephemeris_specificity_and_prefilter.md`
- ☐ `src/bittr_tess_vetter/api/primitives.py`
- ☐ `src/bittr_tess_vetter/api/recovery_primitives.py`
- ☐ `src/bittr_tess_vetter/api/references.py`
- ✅ `src/bittr_tess_vetter/api/report.py` → `working_docs/physics_audit/modules/completed/api_evidence_and_report.md`
- ☐ `src/bittr_tess_vetter/api/sandbox_primitives.py`
- ✅ `src/bittr_tess_vetter/api/systematics.py` → `working_docs/physics_audit/modules/completed/api_systematics_proxy.md`
- ✅ `src/bittr_tess_vetter/api/target.py` → `working_docs/physics_audit/modules/completed/api_io_and_target.md`
- ☐ `src/bittr_tess_vetter/api/timing_primitives.py`
- ✅ `src/bittr_tess_vetter/api/tolerances.py` → `working_docs/physics_audit/modules/completed/utils_and_api_tolerances.md`
- ✅ `src/bittr_tess_vetter/api/tpf.py` → `working_docs/physics_audit/modules/completed/api_tpf_and_tpf_fits.md`
- ✅ `src/bittr_tess_vetter/api/tpf_fits.py` → `working_docs/physics_audit/modules/completed/api_tpf_and_tpf_fits.md`
- ✅ `src/bittr_tess_vetter/api/transit_fit_primitives.py` → `working_docs/physics_audit/modules/completed/api_transit_fit_primitives.md`
- ✅ `src/bittr_tess_vetter/api/transit_primitives.py` → `working_docs/physics_audit/modules/completed/api_transit_primitives.md`
- ☐ `src/bittr_tess_vetter/api/triceratops_cache.py`
- ✅ `src/bittr_tess_vetter/api/vet.py` → `working_docs/physics_audit/modules/completed/api_vet_and_vetting_primitives.md`
- ✅ `src/bittr_tess_vetter/api/vetting_primitives.py` → `working_docs/physics_audit/modules/completed/api_vet_and_vetting_primitives.md`
- ✅ `src/bittr_tess_vetter/api/wcs_utils.py` → `working_docs/physics_audit/modules/completed/api_wcs_utils.md`

### 4.2) Compute layer (physics + numerics implementations)

- ✅ `src/bittr_tess_vetter/compute/aperture_prediction.py` → `working_docs/physics_audit/modules/completed/api_pixel_prf.md`
- ✅ `src/bittr_tess_vetter/compute/bls_like_search.py` → `working_docs/physics_audit/modules/completed/compute_bls_like_search.md`
- ✅ `src/bittr_tess_vetter/compute/detrend.py` → `working_docs/physics_audit/modules/completed/compute_detrend.md`
- ✅ `src/bittr_tess_vetter/compute/joint_inference_schemas.py` → `working_docs/physics_audit/modules/completed/api_pixel_prf.md`
- ✅ `src/bittr_tess_vetter/compute/joint_likelihood.py` → `working_docs/physics_audit/modules/completed/api_pixel_prf.md`
- ✅ `src/bittr_tess_vetter/compute/mlx_detection.py` → `working_docs/physics_audit/modules/completed/api_mlx_and_compute_mlx_detection.md`
- ✅ `src/bittr_tess_vetter/compute/model_competition.py` → `working_docs/physics_audit/modules/completed/model_competition.md`
- ✅ `src/bittr_tess_vetter/compute/periodogram.py` → `working_docs/physics_audit/modules/completed/api_periodogram.md`
- ✅ `src/bittr_tess_vetter/compute/pixel_host_hypotheses.py` → `working_docs/physics_audit/modules/completed/api_pixel_prf.md`
- ✅ `src/bittr_tess_vetter/compute/pixel_hypothesis_prf.py` → `working_docs/physics_audit/modules/completed/api_pixel_prf.md`
- ✅ `src/bittr_tess_vetter/compute/pixel_prf_lite.py` → `working_docs/physics_audit/modules/completed/api_pixel_prf.md`
- ✅ `src/bittr_tess_vetter/compute/pixel_timeseries.py` → `working_docs/physics_audit/modules/completed/api_pixel_prf.md`
- ✅ `src/bittr_tess_vetter/compute/prf_psf.py` → `working_docs/physics_audit/modules/completed/api_pixel_prf.md`
- ✅ `src/bittr_tess_vetter/compute/prf_schemas.py` → `working_docs/physics_audit/modules/completed/api_pixel_prf.md`
- ✅ `src/bittr_tess_vetter/compute/mlx_detection.py` → `working_docs/physics_audit/modules/completed/api_mlx_and_compute_mlx_detection.md`
- ✅ `src/bittr_tess_vetter/compute/primitives.py` → `working_docs/physics_audit/modules/completed/compute_primitives.md`
- ✅ `src/bittr_tess_vetter/compute/transit.py` → `working_docs/physics_audit/modules/completed/compute_transit_and_recovery_pipeline.md`

### 4.3) Catalogs + crossmatch (astro context + priors)

- ✅ `src/bittr_tess_vetter/catalogs/crossmatch.py` → `working_docs/physics_audit/modules/completed/catalogs_spatial_and_crossmatch.md`
- ✅ `src/bittr_tess_vetter/catalogs/exofop_target_page.py` → `working_docs/physics_audit/modules/completed/catalogs_exofop_target_page_and_toi_table.md`
- ✅ `src/bittr_tess_vetter/catalogs/exofop_toi_table.py` → `working_docs/physics_audit/modules/completed/catalogs_exofop_target_page_and_toi_table.md`
- ✅ `src/bittr_tess_vetter/catalogs/exoplanet_archive.py` → `working_docs/physics_audit/modules/completed/catalogs_snapshot_id_and_exoplanet_archive.md`
- ✅ `src/bittr_tess_vetter/catalogs/gaia_client.py` → `working_docs/physics_audit/modules/completed/catalogs_gaia_simbad_store.md`
- ✅ `src/bittr_tess_vetter/catalogs/simbad_client.py` → `working_docs/physics_audit/modules/completed/catalogs_gaia_simbad_store.md`
- ✅ `src/bittr_tess_vetter/catalogs/snapshot_id.py` → `working_docs/physics_audit/modules/completed/catalogs_snapshot_id_and_exoplanet_archive.md`
- ✅ `src/bittr_tess_vetter/catalogs/spatial.py` → `working_docs/physics_audit/modules/completed/catalogs_spatial_and_crossmatch.md`
- ✅ `src/bittr_tess_vetter/catalogs/store.py` → `working_docs/physics_audit/modules/completed/catalogs_gaia_simbad_store.md`

### 4.4) Pixel pipeline (non-API package mirrors)

- ✅ `src/bittr_tess_vetter/pixel/aperture.py` → `working_docs/physics_audit/modules/completed/api_aperture_dependence.md`
- ✅ `src/bittr_tess_vetter/pixel/aperture_family.py` → `working_docs/physics_audit/modules/completed/api_aperture_family.md`
- ✅ `src/bittr_tess_vetter/pixel/cadence_mask.py` → `working_docs/physics_audit/modules/completed/pixel_cadence_mask.md`
- ✅ `src/bittr_tess_vetter/pixel/centroid.py` → `working_docs/physics_audit/modules/completed/api_centroid_shift.md`
- ✅ `src/bittr_tess_vetter/pixel/difference.py` → `working_docs/physics_audit/modules/completed/api_proxy_localization.md`
- ✅ `src/bittr_tess_vetter/pixel/localization.py` → `working_docs/physics_audit/modules/completed/api_proxy_localization.md`
- ✅ `src/bittr_tess_vetter/pixel/report.py` → `working_docs/physics_audit/modules/completed/pixel_report.md`
- ✅ `src/bittr_tess_vetter/pixel/tpf.py` → `working_docs/physics_audit/modules/completed/domain_target_and_pixel_tpf.md`
- ✅ `src/bittr_tess_vetter/pixel/tpf_fits.py` → `working_docs/physics_audit/modules/completed/domain_target_and_pixel_tpf.md`
- ✅ `src/bittr_tess_vetter/pixel/wcs_localization.py` → `working_docs/physics_audit/modules/completed/api_wcs_localization.md`
- ✅ `src/bittr_tess_vetter/pixel/wcs_utils.py` → `working_docs/physics_audit/modules/completed/api_wcs_utils.md`

### 4.5) Transit + recovery packages (non-API package mirrors)

- ✅ `src/bittr_tess_vetter/recovery/pipeline.py` → `working_docs/physics_audit/modules/completed/compute_transit_and_recovery_pipeline.md`
- ✅ `src/bittr_tess_vetter/recovery/primitives.py` → `working_docs/physics_audit/modules/completed/recovery_primitives_and_result.md`
- ✅ `src/bittr_tess_vetter/recovery/result.py` → `working_docs/physics_audit/modules/completed/recovery_primitives_and_result.md`
- ✅ `src/bittr_tess_vetter/transit/result.py` → `working_docs/physics_audit/modules/completed/transit_timing_and_result.md`
- ✅ `src/bittr_tess_vetter/transit/timing.py` → `working_docs/physics_audit/modules/completed/transit_timing_and_result.md`
- ✅ `src/bittr_tess_vetter/transit/vetting.py` → `working_docs/physics_audit/modules/completed/api_vet_and_vetting_primitives.md`

### 4.6) Validation (additional modules)

- ✅ `src/bittr_tess_vetter/validation/checks_catalog.py` → `working_docs/physics_audit/modules/completed/validation_checks_pixel_and_catalog.md`
- ✅ `src/bittr_tess_vetter/validation/checks_pixel.py` → `working_docs/physics_audit/modules/completed/validation_checks_pixel_and_catalog.md`
- ✅ `src/bittr_tess_vetter/validation/ephemeris_specificity.py` → `working_docs/physics_audit/modules/completed/validation_ephemeris_specificity_and_prefilter.md`
- ✅ `src/bittr_tess_vetter/validation/prefilter.py` → `working_docs/physics_audit/modules/completed/validation_ephemeris_specificity_and_prefilter.md`
- ✅ `src/bittr_tess_vetter/validation/systematics_proxy.py` → `working_docs/physics_audit/modules/completed/api_systematics_proxy.md`
- ✅ `src/bittr_tess_vetter/validation/triceratops_fpp.py` → `working_docs/physics_audit/modules/completed/api_fpp.md`

### 4.7) Supporting domain / utils / IO

- ✅ `src/bittr_tess_vetter/domain/detection.py` → `working_docs/physics_audit/modules/completed/compute_transit_and_recovery_pipeline.md`
- ✅ `src/bittr_tess_vetter/domain/lightcurve.py` → `working_docs/physics_audit/modules/completed/api_lightcurve.md`
- ✅ `src/bittr_tess_vetter/domain/target.py` → `working_docs/physics_audit/modules/completed/domain_target_and_pixel_tpf.md`
- ✅ `src/bittr_tess_vetter/activity/primitives.py` → `working_docs/physics_audit/modules/completed/activity_primitives_and_result.md`
- ✅ `src/bittr_tess_vetter/activity/result.py` → `working_docs/physics_audit/modules/completed/activity_primitives_and_result.md`
- ✅ `src/bittr_tess_vetter/io/cache.py` → `working_docs/physics_audit/modules/completed/io_cache.md`
- ✅ `src/bittr_tess_vetter/io/mast_client.py` → `working_docs/physics_audit/modules/completed/api_lightcurve.md`
- ☐ `src/bittr_tess_vetter/network/timeout.py`
- ☐ `src/bittr_tess_vetter/utils/canonical.py`
- ☐ `src/bittr_tess_vetter/utils/caps.py`
- ✅ `src/bittr_tess_vetter/utils/tolerances.py` → `working_docs/physics_audit/modules/completed/utils_and_api_tolerances.md`
- ☐ `src/bittr_tess_vetter/errors.py`

### 4.8) Vendored (third-party) — optional but recommended

- ☐ `src/bittr_tess_vetter/ext/triceratops_plus_vendor/triceratops/funcs.py`
- ☐ `src/bittr_tess_vetter/ext/triceratops_plus_vendor/triceratops/get_apertures.py`
- ☐ `src/bittr_tess_vetter/ext/triceratops_plus_vendor/triceratops/likelihoods.py`
- ☐ `src/bittr_tess_vetter/ext/triceratops_plus_vendor/triceratops/marginal_likelihoods.py`
- ☐ `src/bittr_tess_vetter/ext/triceratops_plus_vendor/triceratops/priors.py`
- ☐ `src/bittr_tess_vetter/ext/triceratops_plus_vendor/triceratops/triceratops.py`

## Status legend

- ☐ Not started
- ◐ In progress
- ✅ Done
- ⚠️ Needs follow-up
