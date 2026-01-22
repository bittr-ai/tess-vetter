"""Feature builder: transforms raw evidence into ML features."""

from dataclasses import asdict
from typing import Any

from .aggregates import (
    Aggregates,
    CheckPresenceFlags,
    GhostSectorInput,
    HostPlausibilityInput,
    HostScenario,
    LocalizationInput,
    PixelHostInput,
    PixelTimeseriesInput,
    V09Metrics,
    build_aggregates,
)
from .config import FeatureConfig
from .evidence import RawEvidencePacket, is_skip_block
from .schema import FEATURE_SCHEMA_VERSION, EnrichedRow

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def _get_check_metrics(check_results: list[dict[str, Any]], check_id: str) -> dict[str, Any]:
    """Extract metrics dict from check results by check ID."""
    for item in check_results:
        if item.get("id") == check_id:
            metrics = item.get("metrics")
            return metrics if isinstance(metrics, dict) else {}
    return {}


def _as_bool(v: Any) -> bool | None:
    """Convert value to boolean, returning None if not convertible."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "t", "yes", "y", "1"}:
            return True
        if s in {"false", "f", "no", "n", "0"}:
            return False
    return None


def _as_float(x: Any) -> float | None:
    """Convert value to float, returning None if not convertible."""
    if x is None:
        return None
    try:
        f = float(x)
        # Check for finite (not inf/nan)
        if f != f or f == float("inf") or f == float("-inf"):  # noqa: PLR0124
            return None
        return f
    except Exception:
        return None


def _as_int(x: Any) -> int | None:
    """Convert value to int, returning None if not convertible."""
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


def _extract_ghost_sectors(
    pixel_host_hypotheses: dict[str, Any] | None,
) -> list[GhostSectorInput] | None:
    """Extract ghost sector inputs from pixel_host_hypotheses."""
    if not isinstance(pixel_host_hypotheses, dict) or is_skip_block(pixel_host_hypotheses):
        return None
    ghost_summary = pixel_host_hypotheses.get("ghost_summary_by_sector")
    if not isinstance(ghost_summary, list) or not ghost_summary:
        return None
    result: list[GhostSectorInput] = []
    for row in ghost_summary:
        if not isinstance(row, dict):
            continue
        entry: GhostSectorInput = {}
        sector = row.get("sector")
        if sector is not None:
            entry["sector"] = int(sector)
        gs = _as_float(row.get("ghost_like_score_adjusted"))
        if gs is not None:
            entry["ghost_like_score_adjusted"] = gs
        sr = _as_float(row.get("scattered_light_risk"))
        if sr is not None:
            entry["scattered_light_risk"] = sr
        asc = row.get("aperture_sign_consistent")
        if asc is not None:
            entry["aperture_sign_consistent"] = bool(asc)
        if entry:
            result.append(entry)
    return result if result else None


def _extract_localization(
    localization: dict[str, Any] | None,
) -> LocalizationInput | None:
    """Extract localization input from top-level localization dict."""
    if not isinstance(localization, dict) or is_skip_block(localization):
        return None
    result: LocalizationInput = {}
    verdict = localization.get("verdict")
    if verdict is not None:
        result["verdict"] = verdict
    target_dist = _as_float(localization.get("target_distance_arcsec"))
    if target_dist is not None:
        result["target_distance_arcsec"] = target_dist
    uncertainty = _as_float(localization.get("uncertainty_semimajor_arcsec"))
    if uncertainty is not None:
        result["uncertainty_semimajor_arcsec"] = uncertainty
    host_ambig = localization.get("host_ambiguous_within_1pix")
    if host_ambig is not None:
        result["host_ambiguous_within_1pix"] = bool(host_ambig)
    warnings = localization.get("warnings")
    if isinstance(warnings, list):
        result["warnings"] = [str(w) for w in warnings]
    return result if result else None


def _extract_v09_metrics(check_results: list[dict[str, Any]]) -> V09Metrics | None:
    """Extract V09 metrics from check results."""
    v09 = _get_check_metrics(check_results, "V09")
    if not v09:
        return None
    result: V09Metrics = {}
    dist_px = _as_float(v09.get("distance_to_target_pixels"))
    if dist_px is not None:
        result["distance_to_target_pixels"] = dist_px
    loc_rel = _as_bool(v09.get("localization_reliable"))
    if loc_rel is not None:
        result["localization_reliable"] = loc_rel
    warnings = v09.get("warnings")
    if isinstance(warnings, list):
        result["warnings"] = [str(w) for w in warnings]
    return result if result else None


def _extract_pixel_host(
    pixel_host_hypotheses: dict[str, Any] | None,
    check_results: list[dict[str, Any]],
) -> PixelHostInput | None:
    """Extract pixel host input from pixel_host_hypotheses and check results."""
    if not isinstance(pixel_host_hypotheses, dict) or is_skip_block(pixel_host_hypotheses):
        return None
    result: PixelHostInput = {}

    # Ambiguity / disagreement flags
    ambiguity = pixel_host_hypotheses.get("ambiguity")
    if ambiguity is not None:
        result["ambiguity"] = str(ambiguity)
    disagreement = pixel_host_hypotheses.get("disagreement_flag")
    if disagreement is not None:
        result["disagreement_flag"] = str(disagreement)
    flip_rate = _as_float(pixel_host_hypotheses.get("flip_rate"))
    if flip_rate is not None:
        result["flip_rate"] = flip_rate

    # Timeseries data
    ts_verdict = pixel_host_hypotheses.get("timeseries_verdict") or pixel_host_hypotheses.get(
        "pixel_timeseries_verdict"
    )
    ts_delta_raw = pixel_host_hypotheses.get("timeseries_delta_chi2")
    if ts_delta_raw is None:
        ts_delta_raw = pixel_host_hypotheses.get("pixel_timeseries_delta_chi2")
    ts_delta = _as_float(ts_delta_raw)
    ts_best = pixel_host_hypotheses.get("timeseries_best_source_id")
    ts_n_windows = _as_int(pixel_host_hypotheses.get("timeseries_n_windows"))
    ts_agrees = pixel_host_hypotheses.get("timeseries_agrees_with_consensus")

    if any(v is not None for v in [ts_verdict, ts_delta, ts_best, ts_n_windows, ts_agrees]):
        ts_input: PixelTimeseriesInput = {}
        if ts_verdict is not None:
            ts_input["verdict"] = ts_verdict
        if ts_delta is not None:
            ts_input["delta_chi2"] = ts_delta
        if ts_best is not None:
            ts_input["best_source_id"] = str(ts_best)
        if ts_n_windows is not None:
            ts_input["n_windows"] = ts_n_windows
        if ts_agrees is not None:
            ts_input["agrees_with_consensus"] = bool(ts_agrees)
        result["timeseries"] = ts_input

    # Ghost by sector
    ghost_sectors = _extract_ghost_sectors(pixel_host_hypotheses)
    if ghost_sectors:
        result["ghost_by_sector"] = ghost_sectors

    # Host plausibility
    host_plaus = _extract_host_plausibility_from_auto(
        pixel_host_hypotheses.get("host_plausibility_auto")
    )
    if host_plaus:
        result["host_plausibility"] = host_plaus

    return result if result else None


def _extract_host_plausibility_from_auto(
    host_plausibility_auto: dict[str, Any] | None,
) -> HostPlausibilityInput | None:
    """Extract host plausibility input from host_plausibility_auto dict."""
    if not isinstance(host_plausibility_auto, dict):
        return None
    result: HostPlausibilityInput = {}

    # Physics flags
    physics_flags = host_plausibility_auto.get("physics_flags")
    if isinstance(physics_flags, dict):
        requires_followup = _as_bool(physics_flags.get("requires_resolved_followup"))
        if requires_followup is not None:
            result["requires_resolved_followup"] = requires_followup

    rationale = host_plausibility_auto.get("rationale")
    if rationale is None and isinstance(physics_flags, dict):
        rationale = physics_flags.get("rationale")
    if rationale is not None:
        result["rationale"] = str(rationale)

    # Physically impossible source IDs
    impossible_ids = host_plausibility_auto.get("physically_impossible_host_source_ids")
    if impossible_ids is None:
        impossible_ids = host_plausibility_auto.get("physically_impossible_source_ids")
    if isinstance(impossible_ids, list):
        result["physically_impossible_source_ids"] = [str(sid) for sid in impossible_ids]

    # Scenarios
    scenarios_raw = host_plausibility_auto.get("scenarios")
    if isinstance(scenarios_raw, list):
        scenarios: list[HostScenario] = []
        impossible_set = set(result.get("physically_impossible_source_ids") or [])
        for sc in scenarios_raw:
            if not isinstance(sc, dict):
                continue
            scenario: HostScenario = {}
            # Support both legacy layout (scenario.host.source_id) and the newer
            # flat layout used by our enrichment pipeline (scenario.source_id).
            host = sc.get("host")
            if isinstance(host, dict):
                source_id = host.get("source_id")
            else:
                source_id = sc.get("source_id")
            if source_id is not None:
                scenario["source_id"] = str(source_id)
                scenario["physically_impossible"] = bool(
                    sc.get("physically_impossible") or (str(source_id) in impossible_set)
                )
            flux_frac = _as_float(sc.get("flux_fraction"))
            if flux_frac is not None:
                scenario["flux_fraction"] = flux_frac
            true_depth = _as_float(sc.get("true_depth_ppm"))
            if true_depth is not None:
                scenario["true_depth_ppm"] = true_depth
            dcf = _as_float(sc.get("depth_correction_factor"))
            if dcf is not None:
                scenario["depth_correction_factor"] = dcf
            if scenario:
                scenarios.append(scenario)
        if scenarios:
            result["scenarios"] = scenarios

    return result if result else None


def _extract_host_plausibility(
    check_results: list[dict[str, Any]],
) -> HostPlausibilityInput | None:
    """Extract host plausibility from check results (e.g., V13 if available)."""
    # Note: Host plausibility typically comes from pixel_host_hypotheses, not check results
    # This function is provided for completeness if check-based extraction is needed
    # Currently returns None as host plausibility is extracted from pixel_host_hypotheses
    return None


def _build_presence_flags(
    raw: RawEvidencePacket,
    check_results: list[dict[str, Any]],
) -> CheckPresenceFlags:
    """Build presence flags from raw evidence packet."""
    flags: CheckPresenceFlags = {}

    pixel_host = raw.get("pixel_host_hypotheses")
    localization = raw.get("localization")

    # TPF is present if we have any pixel-level analysis
    has_tpf = isinstance(pixel_host, dict) or isinstance(localization, dict)
    flags["has_tpf"] = has_tpf

    # Localization presence
    flags["has_localization"] = isinstance(localization, dict)

    # Diff image presence (check V09 metrics)
    v09 = _get_check_metrics(check_results, "V09")
    flags["has_diff_image"] = _as_float(v09.get("distance_to_target_pixels")) is not None

    # Aperture family presence (check V08 or V10)
    v08 = _get_check_metrics(check_results, "V08")
    v10 = _get_check_metrics(check_results, "V10")
    flags["has_aperture_family"] = bool(v08) or bool(v10)

    # Pixel timeseries presence
    has_ts = False
    if isinstance(pixel_host, dict):
        has_ts = (
            pixel_host.get("timeseries_verdict") is not None
            or pixel_host.get("pixel_timeseries_verdict") is not None
        )
    flags["has_pixel_timeseries"] = has_ts

    # Ghost summary presence
    has_ghost = False
    if isinstance(pixel_host, dict):
        ghost_summary = pixel_host.get("ghost_summary_by_sector")
        has_ghost = isinstance(ghost_summary, list) and len(ghost_summary) > 0
    flags["has_ghost_summary"] = has_ghost

    # Host plausibility presence
    has_host = False
    if isinstance(pixel_host, dict):
        has_host = pixel_host.get("host_plausibility_auto") is not None
    flags["has_host_plausibility"] = has_host

    return flags


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------


def build_features(
    raw: RawEvidencePacket,
    config: FeatureConfig,
) -> EnrichedRow:
    """
    Convert raw check outputs into ML features.

    This function is the single entry point for feature extraction. It takes
    raw evidence from the vetting pipeline and transforms it into a structured
    feature row suitable for ML model input.

    The transformation is **deterministic**: given the same raw evidence packet
    and config, the output will always be identical. Any change to feature
    semantics requires bumping FEATURE_SCHEMA_VERSION.

    Parameters
    ----------
    raw : RawEvidencePacket
        Raw outputs from vetting checks including check results,
        pixel analysis, localization, and candidate evidence.
    config : FeatureConfig
        Configuration controlling which feature families to extract
        and pipeline behavior settings.

    Returns
    -------
    EnrichedRow
        Complete feature row with all extracted features. Missing
        features are set to None and the corresponding family is
        added to `missing_feature_families`.

    Notes
    -----
    Feature extraction follows these principles:

    1. **Determinism**: Same inputs always produce same outputs
    2. **Explicit missingness**: None values with tracked families
    3. **Version tracking**: Schema version embedded in every row
    4. **Config preservation**: Full config dict stored for reproducibility
    """
    # Extract top-level data
    target = raw.get("target") or {}
    ephemeris = raw.get("ephemeris") or {}
    depth_info = raw.get("depth_ppm") or {}
    check_results = raw.get("check_results") or []
    pixel_host_hypotheses = raw.get("pixel_host_hypotheses")
    localization = raw.get("localization")
    candidate_evidence = raw.get("candidate_evidence")
    provenance = raw.get("provenance") or {}

    # Extract check metrics by ID
    pf01 = _get_check_metrics(check_results, "PF01")
    v01 = _get_check_metrics(check_results, "V01")
    v02 = _get_check_metrics(check_results, "V02")
    v03 = _get_check_metrics(check_results, "V03")
    v04 = _get_check_metrics(check_results, "V04")
    v05 = _get_check_metrics(check_results, "V05")
    v08 = _get_check_metrics(check_results, "V08")
    v09 = _get_check_metrics(check_results, "V09")
    v11 = _get_check_metrics(check_results, "V11")
    v11b = _get_check_metrics(check_results, "V11b")

    # Track missing feature families
    missing_feature_families: list[str] = []

    # -------------------------------------------------------------------------
    # Required fields
    # -------------------------------------------------------------------------
    tic_id = target.get("tic_id")
    if tic_id is None:
        msg = "target.tic_id is required"
        raise ValueError(msg)
    tic_id = int(tic_id)

    toi = target.get("toi")
    period_days = ephemeris.get("period_days")
    if period_days is None:
        msg = "ephemeris.period_days is required"
        raise ValueError(msg)
    period_days = float(period_days)

    t0_btjd = ephemeris.get("t0_btjd")
    if t0_btjd is None:
        msg = "ephemeris.t0_btjd is required"
        raise ValueError(msg)
    t0_btjd = float(t0_btjd)

    duration_hours = ephemeris.get("duration_hours")
    if duration_hours is None:
        msg = "ephemeris.duration_hours is required"
        raise ValueError(msg)
    duration_hours = float(duration_hours)

    # Depth is optional but, when provided, must round-trip from the worklist.
    # Our RawEvidencePacket stores it as input_depth_ppm (legacy adapters used "value").
    depth_ppm = _as_float(depth_info.get("input_depth_ppm") or depth_info.get("value"))

    # Build canonical candidate key (stable; independent of any refinement).
    candidate_key = f"{tic_id}|{period_days}|{t0_btjd}"

    # -------------------------------------------------------------------------
    # SNR / Depth proxies
    # -------------------------------------------------------------------------
    snr = _as_float(pf01.get("snr"))
    snr_proxy = _as_float(pf01.get("snr_proxy"))
    depth_est_ppm = _as_float(pf01.get("depth_est_ppm"))
    n_in_transit = _as_int(pf01.get("n_in_transit"))
    n_out_of_transit = _as_int(pf01.get("n_out_of_transit"))

    # -------------------------------------------------------------------------
    # V01: Odd/Even Transit Analysis
    # -------------------------------------------------------------------------
    # V01 uses delta_sigma (or depth_diff_sigma in older versions)
    odd_even_sigma = _as_float(v01.get("delta_sigma") or v01.get("depth_diff_sigma"))

    # V01 returns rel_diff as fraction, convert to percent
    odd_even_rel_diff = v01.get("rel_diff")
    odd_even_relative_diff_percent: float | None = None
    if odd_even_rel_diff is not None:
        odd_even_relative_diff_percent = _as_float(odd_even_rel_diff)
        if odd_even_relative_diff_percent is not None:
            odd_even_relative_diff_percent = odd_even_relative_diff_percent * 100.0

    # Fallback heuristic for is_suspicious
    odd_even_is_suspicious: bool | None = None
    if odd_even_relative_diff_percent is not None:
        odd_even_is_suspicious = bool(odd_even_relative_diff_percent >= 10.0)

    if odd_even_sigma is None:
        missing_feature_families.append("ODD_EVEN")

    # -------------------------------------------------------------------------
    # V02: Secondary Eclipse Analysis
    # -------------------------------------------------------------------------
    secondary_depth_sigma = _as_float(v02.get("secondary_depth_sigma"))
    secondary_significant = _as_bool(v02.get("significant_secondary"))
    if secondary_significant is None and secondary_depth_sigma is not None:
        # V02 emits the continuous `secondary_depth_sigma` metric but not a boolean.
        # Use a conservative threshold consistent with "significant" language.
        secondary_significant = bool(secondary_depth_sigma >= 3.0)

    if secondary_depth_sigma is None:
        missing_feature_families.append("SECONDARY")

    # -------------------------------------------------------------------------
    # V03: Duration Ratio
    # -------------------------------------------------------------------------
    duration_ratio = _as_float(v03.get("duration_ratio"))

    if duration_ratio is None:
        missing_feature_families.append("DURATION")

    # -------------------------------------------------------------------------
    # V04: Depth Scatter / Consistency
    # -------------------------------------------------------------------------
    depth_rms_scatter = _as_float(v04.get("rms_scatter"))
    v04_dmm = _as_float(v04.get("dmm"))
    v04_dom_ratio = _as_float(v04.get("dom_ratio"))

    if depth_rms_scatter is None and v04_dmm is None:
        missing_feature_families.append("DEPTH_SCATTER")

    # -------------------------------------------------------------------------
    # V05: Transit Shape Analysis
    # -------------------------------------------------------------------------
    transit_shape_ratio = _as_float(v05.get("shape_ratio"))
    transit_shape_raw = v05.get("shape")
    transit_shape: str | None = str(transit_shape_raw) if isinstance(transit_shape_raw, str) else None
    # V05 currently provides continuous shape metrics but no discrete label; derive one.
    if transit_shape is None:
        t_flat = _as_float(v05.get("t_flat_hours"))
        t_total = _as_float(v05.get("t_total_hours"))
        ratio = None
        if t_flat is not None and t_total is not None and t_total > 0:
            ratio = t_flat / t_total
        # Fallback to legacy shape_ratio (depth_bottom/depth_edge); U-shape tends to have >1.
        if ratio is None and transit_shape_ratio is not None:
            # Map to an approximate flatness proxy in [0,1] (heuristic).
            ratio = max(0.0, min(1.0, 1.0 - 1.0 / max(1.0, transit_shape_ratio)))
        if ratio is not None:
            if ratio <= 0.10:
                transit_shape = "V"
            elif ratio >= 0.30:
                transit_shape = "U"
            else:
                transit_shape = "AMBIG"

    if transit_shape_ratio is None:
        missing_feature_families.append("TRANSIT_SHAPE")

    # -------------------------------------------------------------------------
    # V11: ModShift Analysis (exovetter)
    # -------------------------------------------------------------------------
    modshift_secondary_ratio = _as_float(v11.get("secondary_primary_ratio"))
    modshift_fred = _as_float(v11.get("fred"))

    # Derive modshift_significant_secondary from V11 ratio > 0.5
    modshift_significant_secondary: bool | None = None
    if modshift_secondary_ratio is not None:
        modshift_significant_secondary = modshift_secondary_ratio > 0.5

    if modshift_secondary_ratio is None and modshift_fred is None:
        missing_feature_families.append("MODSHIFT")

    # -------------------------------------------------------------------------
    # V11b: ModShift TESS (independent impl)
    # -------------------------------------------------------------------------
    v11b_sig_pri = _as_float(v11b.get("sig_pri"))
    v11b_sig_sec = _as_float(v11b.get("sig_sec"))
    v11b_fred = _as_float(v11b.get("fred"))

    if v11b_sig_pri is None:
        missing_feature_families.append("MODSHIFT_TESS")

    # -------------------------------------------------------------------------
    # V08: Centroid Shift
    # -------------------------------------------------------------------------
    centroid_shift_pixels = _as_float(v08.get("centroid_shift_pixels"))

    # -------------------------------------------------------------------------
    # V09: Difference Image Localization
    # -------------------------------------------------------------------------
    diff_image_distance_to_target_pixels = _as_float(v09.get("distance_to_target_pixels"))

    # -------------------------------------------------------------------------
    # Build aggregates via the aggregates subpackage
    # -------------------------------------------------------------------------
    # Extract minimal contract inputs
    ghost_sectors = _extract_ghost_sectors(pixel_host_hypotheses)
    localization_input = _extract_localization(localization)
    v09_metrics = _extract_v09_metrics(check_results)
    pixel_host_input = _extract_pixel_host(pixel_host_hypotheses, check_results)
    host_plausibility_input = (
        pixel_host_input.get("host_plausibility") if pixel_host_input else None
    )
    presence_flags = _build_presence_flags(raw, check_results)

    # Call build_aggregates to compute all aggregated features
    aggregates: Aggregates = build_aggregates(
        ghost_sectors=ghost_sectors,
        localization=localization_input,
        v09=v09_metrics,
        pixel_host=pixel_host_input,
        host_plausibility=host_plausibility_input,
        presence_flags=presence_flags,
    )

    # Extract results from aggregates
    ghost_summary = aggregates.get("ghost", {})
    localization_summary = aggregates.get("localization", {})
    host_plausibility_summary = aggregates.get("host_plausibility", {})
    pixel_host_summary = aggregates.get("pixel_host", {})
    coverage_summary = aggregates.get("coverage", {})

    # Ghost / Aperture features
    ghost_like_score_adjusted_median = ghost_summary.get("ghost_like_score_adjusted_median")
    scattered_light_risk_median = ghost_summary.get("scattered_light_risk_median")
    aperture_sign_consistent_all = ghost_summary.get("aperture_sign_consistent_all")

    # Localization features
    localization_verdict = localization_summary.get("localization_verdict")

    # Pixel timeseries features
    pixel_timeseries_verdict = pixel_host_summary.get("pixel_timeseries_verdict")
    pixel_timeseries_delta_chi2 = pixel_host_summary.get("pixel_timeseries_delta_chi2")

    # Host plausibility features
    host_requires_resolved_followup = host_plausibility_summary.get(
        "host_requires_resolved_followup"
    )
    host_physically_impossible_count = host_plausibility_summary.get(
        "host_physically_impossible_count"
    )
    # Convert source_id from str to int for schema compatibility
    _source_id_str = host_plausibility_summary.get("host_feasible_best_source_id")
    host_feasible_best_source_id: int | None = (
        _as_int(_source_id_str) if _source_id_str is not None else None
    )

    # -------------------------------------------------------------------------
    # Update missing feature families from coverage summary
    # -------------------------------------------------------------------------
    coverage_missing = coverage_summary.get("missing_feature_families") or []
    for family in coverage_missing:
        if family not in missing_feature_families:
            missing_feature_families.append(family)

    # Legacy missing family tracking (for backward compatibility)
    if (
        localization_verdict is None
        and diff_image_distance_to_target_pixels is None
        and "TPF_LOCALIZATION" not in missing_feature_families
    ):
        missing_feature_families.append("TPF_LOCALIZATION")

    if (
        config.enable_pixel_timeseries
        and pixel_timeseries_verdict is None
        and "PIXEL_TIMESERIES" not in missing_feature_families
    ):
        missing_feature_families.append("PIXEL_TIMESERIES")

    if (
        config.enable_ghost_reliability
        and ghost_like_score_adjusted_median is None
        and "GHOST_RELIABILITY" not in missing_feature_families
    ):
        missing_feature_families.append("GHOST_RELIABILITY")

    if (
        config.enable_host_plausibility
        and host_requires_resolved_followup is None
        and "HOST_PLAUSIBILITY" not in missing_feature_families
    ):
        missing_feature_families.append("HOST_PLAUSIBILITY")

    # -------------------------------------------------------------------------
    # Gaia Crowding Metrics (from candidate_evidence)
    # -------------------------------------------------------------------------
    n_gaia_neighbors_21arcsec: int | None = None
    brightest_neighbor_delta_mag: float | None = None
    crowding_metric: float | None = None

    if isinstance(candidate_evidence, dict) and not is_skip_block(candidate_evidence):
        gaia_crowding = candidate_evidence.get("gaia_crowding") or {}
        if isinstance(gaia_crowding, dict):
            n_gaia_neighbors_21arcsec = _as_int(gaia_crowding.get("n_gaia_neighbors_21arcsec"))
            brightest_neighbor_delta_mag = _as_float(
                gaia_crowding.get("brightest_neighbor_delta_mag")
            )
            crowding_metric = _as_float(gaia_crowding.get("crowding_metric"))

    # -------------------------------------------------------------------------
    # Build inputs_summary
    # -------------------------------------------------------------------------
    inputs_summary: dict[str, Any] = {}
    if "sectors" in ephemeris:
        inputs_summary["sectors"] = ephemeris["sectors"]
    if "cadence_seconds" in ephemeris:
        inputs_summary["cadence_seconds"] = ephemeris["cadence_seconds"]
    # Cache/provenance hints (useful for training reproducibility).
    if getattr(config, "cache_dir", None) is not None:
        inputs_summary["cache_dir"] = str(getattr(config, "cache_dir"))
    inputs_summary["cache_only"] = bool(getattr(config, "no_download", False))
    if isinstance(provenance, dict):
        ch = provenance.get("code_hash")
        if isinstance(ch, str) and ch:
            inputs_summary["btv_code_hash"] = ch
        dv = provenance.get("dependency_versions")
        if isinstance(dv, dict):
            inputs_summary["btv_dependency_versions"] = {str(k): str(v) for k, v in dv.items()}

    # -------------------------------------------------------------------------
    # Build the EnrichedRow
    # -------------------------------------------------------------------------
    row: EnrichedRow = {
        # Required fields
        "tic_id": tic_id,
        "toi": toi,
        "period_days": period_days,
        "t0_btjd": t0_btjd,
        "duration_hours": duration_hours,
        "depth_ppm": depth_ppm,
        "status": "OK",
        "error_class": None,
        "error": None,
        "candidate_key": candidate_key,
        "pipeline_version": provenance.get("pipeline_version", "unknown"),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_config": asdict(config),
        "inputs_summary": inputs_summary,
        "missing_feature_families": missing_feature_families,
        "item_wall_ms": 0.0,  # Caller should update this
        # SNR / Depth proxies
        "snr": snr,
        "snr_proxy": snr_proxy,
        "depth_est_ppm": depth_est_ppm,
        "n_in_transit": n_in_transit,
        "n_out_of_transit": n_out_of_transit,
        # Odd/Even
        "odd_even_sigma": odd_even_sigma,
        "odd_even_relative_diff_percent": odd_even_relative_diff_percent,
        "odd_even_is_suspicious": odd_even_is_suspicious,
        # Secondary
        "secondary_significant": secondary_significant,
        "secondary_depth_sigma": secondary_depth_sigma,
        # Duration / Depth
        "duration_ratio": duration_ratio,
        "depth_rms_scatter": depth_rms_scatter,
        "v04_dmm": v04_dmm,
        "v04_dom_ratio": v04_dom_ratio,
        # Transit shape
        "transit_shape_ratio": transit_shape_ratio,
        "transit_shape": transit_shape,
        # ModShift (V11)
        "modshift_secondary_primary_ratio": modshift_secondary_ratio,
        "modshift_significant_secondary": modshift_significant_secondary,
        "modshift_fred": modshift_fred,
        # ModShift TESS (V11b)
        "v11b_sig_pri": v11b_sig_pri,
        "v11b_sig_sec": v11b_sig_sec,
        "v11b_fred": v11b_fred,
        # Pixel Localization
        "centroid_shift_pixels": centroid_shift_pixels,
        "diff_image_distance_to_target_pixels": diff_image_distance_to_target_pixels,
        "localization_verdict": localization_verdict,
        "pixel_timeseries_verdict": pixel_timeseries_verdict,
        "pixel_timeseries_delta_chi2": pixel_timeseries_delta_chi2,
        # Ghost / Aperture
        "ghost_like_score_adjusted_median": ghost_like_score_adjusted_median,
        "scattered_light_risk_median": scattered_light_risk_median,
        "aperture_sign_consistent_all": aperture_sign_consistent_all,
        # Host Plausibility
        "host_requires_resolved_followup": host_requires_resolved_followup,
        "host_physically_impossible_count": host_physically_impossible_count,
        "host_feasible_best_source_id": host_feasible_best_source_id,
        # Gaia Crowding
        "n_gaia_neighbors_21arcsec": n_gaia_neighbors_21arcsec,
        "brightest_neighbor_delta_mag": brightest_neighbor_delta_mag,
        "crowding_metric": crowding_metric,
    }

    return row


__all__ = ["build_features", "FEATURE_SCHEMA_VERSION"]
