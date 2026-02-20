"""Feature schema definition for ML model input."""

from typing import Literal, TypedDict

# Schema version - bump on any semantic change to feature definitions
FEATURE_SCHEMA_VERSION = "6.0.2"


class EnrichedRow(TypedDict, total=False):
    """
    ML-ready feature row with all extracted features.

    This TypedDict defines the complete schema for enriched candidate rows
    used by ML models. All feature fields are optional (total=False) to
    support partial feature extraction when certain families are disabled.

    The schema is designed to be JSON-serializable for easy storage and
    transmission between pipeline stages.

    Version History
    ---------------
    6.0.0 : Initial v6 schema with explicit family grouping
    6.0.2 : Add LC-only diagnostic fields (V09/V10/V13/V15, ephemeris specificity, alias)
    """

    # =========================================================================
    # Required fields (always present)
    # =========================================================================

    # Target identification
    tic_id: int
    """TIC ID of the target star."""

    toi: str | None
    """TOI designation if available (e.g., 'TOI-1234.01')."""

    # Ephemeris parameters
    period_days: float
    """Orbital period in days."""

    t0_btjd: float
    """Transit epoch in BTJD (BJD - 2457000)."""

    duration_hours: float
    """Transit duration in hours."""

    depth_ppm: float | None
    """Transit depth in parts per million."""

    # Pipeline status
    status: Literal["OK", "ERROR"]
    """Pipeline completion status."""

    error_class: str | None
    """Error class name if status is ERROR."""

    error: str | None
    """Error message if status is ERROR."""

    # Metadata
    candidate_key: str
    """Unique candidate identifier: f"{tic_id}|{period_days}|{t0_btjd}"."""

    pipeline_version: str
    """Version of the vetting pipeline used."""

    feature_schema_version: str
    """Version of this feature schema (should match FEATURE_SCHEMA_VERSION)."""

    feature_config: dict
    """FeatureConfig dict used for extraction (for reproducibility)."""

    inputs_summary: dict
    """Summary of input data used (sectors, cadence, etc.)."""

    missing_feature_families: list[str]
    """List of feature families that could not be computed."""

    item_wall_ms: float
    """Wall-clock time in milliseconds to process this item."""

    # =========================================================================
    # SNR / Depth Proxies
    # =========================================================================

    snr: float | None
    """Signal-to-noise ratio of the transit detection."""

    snr_proxy: float | None
    """Proxy SNR computed from depth and scatter."""

    depth_est_ppm: float | None
    """Estimated depth from transit fitting in ppm."""

    n_in_transit: int | None
    """Number of data points inside transit."""

    n_out_of_transit: int | None
    """Number of data points outside transit (for baseline)."""

    # =========================================================================
    # Odd/Even Transit Analysis
    # =========================================================================

    odd_even_sigma: float | None
    """Significance of odd/even depth difference in sigma."""

    odd_even_relative_diff_percent: float | None
    """Relative difference between odd and even depths as percentage."""

    odd_even_is_suspicious: bool | None
    """Flag indicating suspicious odd/even difference (possible EB)."""

    # =========================================================================
    # Secondary Eclipse Analysis
    # =========================================================================

    secondary_significant: bool | None
    """Whether a significant secondary eclipse was detected."""

    secondary_depth_sigma: float | None
    """Significance of secondary eclipse depth in sigma."""

    # =========================================================================
    # Duration / Depth Metrics
    # =========================================================================

    duration_ratio: float | None
    """Ratio of measured to expected duration (from stellar density)."""

    depth_rms_scatter: float | None
    """RMS scatter of individual transit depths."""

    v04_dmm: float | None
    """V04 depth mean metric (consistency measure)."""

    v04_dom_ratio: float | None
    """V04 depth-to-odd/mean ratio."""

    # =========================================================================
    # Transit Shape Analysis
    # =========================================================================

    transit_shape_ratio: float | None
    """Ratio metric characterizing transit shape (V vs U)."""

    transit_shape: str | None
    """Categorical transit shape classification (e.g., 'V', 'U', 'flat')."""

    # =========================================================================
    # ModShift Analysis (V11 / V11b)
    # =========================================================================

    modshift_secondary_primary_ratio: float | None
    """Ratio of secondary to primary modshift signal."""

    modshift_significant_secondary: bool | None
    """Whether modshift detected significant secondary signal."""

    modshift_fred: float | None
    """ModShift FRED statistic (false-alarm rate estimate)."""

    v11b_secondary_primary_ratio: float | None
    """Secondary/primary ratio derived from V11b (sig_sec / sig_pri)."""

    v11b_sig_pri: float | None
    """V11b primary signal significance."""

    v11b_sig_sec: float | None
    """V11b secondary signal significance."""

    v11b_fred: float | None
    """V11b FRED statistic."""

    v11b_sig_ter: float | None
    """V11b tertiary signal significance."""

    v11b_chi: float | None
    """V11b depth-consistency statistic (CHI)."""

    # =========================================================================
    # LC-only systematics / specificity diagnostics
    # =========================================================================

    smooth_score: float | None
    """Smooth-template score at the provided ephemeris (higher is more transit-like)."""

    null_pvalue: float | None
    """One-sided phase-shift null p-value (lower is more ephemeris-specific)."""

    few_point_fraction: float | None
    """Fraction of absolute contribution from the top few points (higher is more suspicious)."""

    systematics_proxy_score: float | None
    """Cheap LC-only systematics risk score in [0,1] (higher is worse)."""

    # =========================================================================
    # Alias / harmonic diagnostics (LC-only)
    # =========================================================================

    alias_class: str | None
    """Alias classification: CLEAN / ALIAS_RISK / STRONG_ALIAS."""

    # =========================================================================
    # Pixel Localization
    # =========================================================================

    centroid_shift_pixels: float | None
    """Centroid shift during transit in pixels."""

    diff_image_distance_to_target_pixels: float | None
    """Distance from difference image centroid to target in pixels."""

    localization_verdict: str | None
    """Overall localization verdict (e.g., 'on_target', 'off_target')."""

    pixel_timeseries_verdict: str | None
    """Pixel timeseries analysis verdict."""

    pixel_timeseries_delta_chi2: float | None
    """Delta chi-squared from pixel timeseries model comparison."""

    # =========================================================================
    # Pixel QA / false-alarm checks (existing vetting checks)
    # =========================================================================

    v09_localization_reliable: bool | None
    """V09 localization reliability flag (difference image not edge-dominated)."""

    v10_aperture_depth_sign_flip: bool | None
    """V10 aperture-depth sign flip flag (depth changes sign with aperture)."""

    v13_missing_frac_max: float | None
    """V13 maximum missing cadence fraction across transit epochs."""

    v13_n_epochs_missing_ge_0p25: int | None
    """V13 number of epochs with missing_frac >= 0.25."""

    v15_asymmetry_sigma: float | None
    """V15 transit-window asymmetry significance (sigma)."""

    # =========================================================================
    # LC metadata / quality
    # =========================================================================

    lc_cadence_seconds: float | None
    """Light curve cadence in seconds (effective, after selection)."""

    lc_n_valid: int | None
    """Number of valid cadences used in the stitched light curve."""

    pixel_flip_rate: float | None
    """Rate at which pixel-host verdict flips across controls/windows (lower is better)."""

    # =========================================================================
    # Ghost / Aperture Analysis
    # =========================================================================

    ghost_like_score_adjusted_median: float | None
    """Median ghost-like score across sectors (adjusted for systematics)."""

    scattered_light_risk_median: float | None
    """Median scattered light contamination risk."""

    aperture_sign_consistent_all: bool | None
    """Whether aperture flux signs are consistent across all pixels."""

    # =========================================================================
    # Host Plausibility
    # =========================================================================

    host_requires_resolved_followup: bool | None
    """Flag indicating resolved follow-up needed to confirm host."""

    host_physically_impossible_count: int | None
    """Count of physically impossible host scenarios."""

    host_feasible_best_source_id: int | None
    """Gaia source_id of the most plausible host star."""

    # =========================================================================
    # Gaia Crowding Metrics
    # =========================================================================

    n_gaia_neighbors_21arcsec: int | None
    """Number of Gaia sources within 21 arcsec of target."""

    brightest_neighbor_delta_mag: float | None
    """Magnitude difference to brightest neighbor."""

    crowding_metric: float | None
    """Combined crowding metric (flux dilution estimate)."""
